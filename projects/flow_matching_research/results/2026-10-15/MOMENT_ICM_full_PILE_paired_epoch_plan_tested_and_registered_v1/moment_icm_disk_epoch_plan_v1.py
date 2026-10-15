"""Disk-backed exact RandomState paired epoch references, without per-item tuples.

Preserve the previous planner's width buckets, dataset/item order, per-width
permutation, final Python-list batch shuffle and partial batches. A single
native NumPy permutation is live at once; mapped output blocks are released.
This plans references only and never certifies that underlying data are ready.
"""
from __future__ import annotations
from bisect import bisect_right
import gc
import hashlib
import json
from pathlib import Path
from scripts.flow_matching.moment_icm_channel_data_v1 import BatchReference


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def positive(value,name,allow_zero=False):
    if isinstance(value,bool) or not isinstance(value,int) or value<(0 if allow_zero else 1):
        raise ValueError(name+' must be an explicit integer in range')
    if value>2**63-1:raise ValueError(name+' exceeds int64 scope')
    return value


def layout(sources,batch_size):
    positive(batch_size,'batch_size')
    if not sources or len({s['dataset_key'] for s in sources})!=len(sources):raise ValueError('Unique complete source identities required')
    if len({s['split'] for s in sources})!=1:raise ValueError('Never mix source train/val/test splits')
    buckets={}
    for index,s in enumerate(sources):
        width=positive(s['width'],'width');n=positive(s['items'],'items',True)
        if width>8:raise ValueError('Declared 1..8 channel buckets required')
        if not isinstance(s['dataset_key'],str) or not s['dataset_key'] or s['split'] not in ['train','val','test','TRAIN','TEST']:
            raise ValueError('Explicit stable dataset and split identities required')
        if n:
            b=buckets.setdefault(width,dict(source_indices=[],item_ends=[],items=0))
            b['source_indices'].append(index);b['items']+=n;b['item_ends'].append(b['items'])
            positive(b['items'],'bucket_items')
    if not buckets:raise ValueError('One complete nonempty epoch required')
    count=0;records=[]
    for width in sorted(buckets):
        b=buckets[width];b['width']=width;b['batch_start']=count;b['batches']=(b['items']+batch_size-1)//batch_size
        count+=b['batches'];b['batch_end']=count;records.append(b)
    return dict(sources=sources,batch_size=batch_size,buckets=records,total_batches=count,
        total_items=sum(b['items'] for b in records),source_split=sources[0]['split'])


def save_int64_blocks(path,source,block_bytes=16*2**20):
    import numpy as np
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve previous epoch plan array')
    n=len(source);mapped=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(n,))
    mapped.flush();offset=mapped.offset;mapped._mmap.close();del mapped
    step=max(1,block_bytes//8);blocks=0
    for start in range(0,n,step):
        end=min(start+step,n)
        mapped=np.memmap(path,mode='r+',dtype=np.int64,offset=offset+start*8,shape=(end-start,))
        try:
            data=np.asarray(source[start:end],dtype=np.int64)
            mapped[:]=data
            if mapped.tobytes()!=data.tobytes():raise ValueError('Every index must retain exact integer identity')
            mapped.flush();blocks+=1
        finally:mapped._mmap.close();del mapped
    return dict(file=path.name,sha256=sha(path),shape=[n],dtype='int64',offset=offset,write_blocks=blocks,
        mapping_released_after_every_block=True)


def build(directory,sources,batch_size,seed,progress=None):
    import numpy as np
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:raise ValueError('Explicit uint32 epoch seed required')
    plan=layout(sources,batch_size);directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    rng=np.random.RandomState(seed);files={};largest_native_bytes=0
    for bucket in plan['buckets']:
        # Keep the original integer-N permutation call, including RNG semantics.
        order=rng.permutation(bucket['items'])
        largest_native_bytes=max(largest_native_bytes,order.nbytes)
        files[str(bucket['width'])]=save_int64_blocks(directory/f"width{bucket['width']}_permutation.npy",order)
        del order;gc.collect()
        if progress:progress(dict(completed_width=bucket['width'],complete_bucket_items=bucket['items']))
    # Same generic Python-list shuffle as the old BatchReference list. Only one
    # integer per BATCH is materialized, never a tuple for each original item.
    batch_order=list(range(plan['total_batches']));rng.shuffle(batch_order)
    order=np.fromiter(batch_order,dtype=np.int64,count=len(batch_order));del batch_order
    batch_file=save_int64_blocks(directory/'batch_order.npy',order);del order;gc.collect()
    state=rng.get_state()
    result=dict(schema_version=1,**plan,seed=seed,permutation_files=files,batch_order_file=batch_file,
        largest_live_native_permutation_bytes=largest_native_bytes,
        every_original_item_once=True,drop_last=False,
        original_RandomState_draw_order_preserved=True,
        final_RNG=dict(algorithm=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4]),
        data_arrays_read_or_model_executed=False,full_training_data_readiness_certified=False)
    (directory/'epoch_plan.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def read_indices(directory,record,start,count):
    import numpy as np
    if not 0<=start<=start+count<=record['shape'][0]:raise IndexError('Complete index block out of range')
    mapped=np.memmap(Path(directory)/record['file'],mode='r',dtype=np.int64,
        offset=record['offset']+start*8,shape=(count,))
    try:return mapped.copy()
    finally:mapped._mmap.close()


class DiskEpochPlan:
    def __init__(self,directory,verify_hashes=True):
        self.directory=Path(directory);self.plan=json.loads((self.directory/'epoch_plan.json').read_text(encoding='utf-8'))
        expected=layout(self.plan['sources'],self.plan['batch_size'])
        for key in ['buckets','total_batches','total_items','source_split']:
            if self.plan[key]!=expected[key]:raise ValueError('Full original source/bucket/batch scope changed')
        if self.plan['drop_last'] or not self.plan['every_original_item_once']:raise ValueError('Every partial/full batch and item required')
        if set(self.plan['permutation_files'])!={str(b['width']) for b in self.plan['buckets']}:
            raise ValueError('Every original width permutation required')
        expected_records=[(self.plan['permutation_files'][str(b['width'])],b['items']) for b in self.plan['buckets']]
        expected_records.append((self.plan['batch_order_file'],self.plan['total_batches']))
        import numpy as np
        for record,count in expected_records:
            path=self.directory/record['file']
            if verify_hashes and sha(path)!=record['sha256']:raise ValueError('Epoch index hash changed')
            with path.open('rb') as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError('Explicit numeric NPY header required')
                offset=f.tell()
            if (shape!=(count,) or dtype!=np.dtype('int64') or fortran or record['shape']!=[count]
                    or record['dtype']!='int64' or record['offset']!=offset or path.stat().st_size!=offset+count*8):
                raise ValueError('Complete native int64 index array header and extent required')
        self.batch_ends=[b['batch_end'] for b in self.plan['buckets']]

    def __len__(self):return self.plan['total_batches']

    def reference(self,index):
        if isinstance(index,bool) or not isinstance(index,int) or not 0<=index<len(self):raise IndexError('Exact complete epoch batch index required')
        original_batch=int(read_indices(self.directory,self.plan['batch_order_file'],index,1)[0])
        which=bisect_right(self.batch_ends,original_batch)
        if which>=len(self.plan['buckets']):raise ValueError('Original width-batch identity invalid')
        bucket=self.plan['buckets'][which];local=original_batch-bucket['batch_start']
        start=local*self.plan['batch_size'];count=min(self.plan['batch_size'],bucket['items']-start)
        values=read_indices(self.directory,self.plan['permutation_files'][str(bucket['width'])],start,count)
        items=[]
        for value in values:
            value=int(value)
            if not 0<=value<bucket['items']:raise ValueError('Original sample identity out of range')
            source_slot=bisect_right(bucket['item_ends'],value)
            origin=bucket['item_ends'][source_slot-1] if source_slot else 0
            items.append((bucket['source_indices'][source_slot],value-origin))
        return BatchReference(bucket['width'],tuple(items))

    def validate_complete(self,block_items=2**20,progress=None):
        """Exact bijection checks, not only sums/hashes or sampled batches."""
        import numpy as np
        positive(block_items,'block_items');checked=[]
        for key,record in [*self.plan['permutation_files'].items(),('batch_order',self.plan['batch_order_file'])]:
            count=record['shape'][0];seen=np.zeros(count,dtype=bool)
            for start in range(0,count,block_items):
                values=read_indices(self.directory,record,start,min(block_items,count-start))
                if np.any(values<0) or np.any(values>=count) or len(np.unique(values))!=len(values) or np.any(seen[values]):
                    raise ValueError('Every original item/batch must occur exactly once')
                seen[values]=True
                if progress:progress(dict(permutation=key,checked_indices=min(start+len(values),count),required=count))
            if not seen.all():raise ValueError('Complete permutation coverage required')
            checked.append(dict(permutation=key,indices=count,exact_bijection=True));del seen;gc.collect()
        return dict(passed=True,records=checked,complete_epoch_items=self.plan['total_items'],
            complete_epoch_batches=self.plan['total_batches'],every_partial_batch_retained=True,
            all_original_source_items_exactly_once=True,data_content_or_model_performance_certified=False)
