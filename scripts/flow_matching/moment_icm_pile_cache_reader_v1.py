"""Bounded read-only model inputs for all four declared PILE cache families.

Only source bindings from an independently audited experiment may certify full
data readiness. This component never fits a model, uses labels, changes a value,
or silently supplies target validity for original anomaly sources.
"""
from __future__ import annotations
from collections import OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
from scripts.flow_matching.moment_icm_channel_data_v1 import channel_groups


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


class BoundedArrays:
    """Read numeric NPY ranges with capped byte mappings, not persistent huge maps."""
    def __init__(self,cache_bytes=64*2**20,block_bytes=2**20):
        if not isinstance(cache_bytes,int) or not isinstance(block_bytes,int) or not 0<block_bytes<=cache_bytes:
            raise ValueError('Explicit positive bounded mapping sizes required')
        self.cache_bytes,self.block_bytes=cache_bytes,block_bytes
        self.headers={};self.maps=OrderedDict();self.mapped_bytes=0;self.peak_mapped_bytes=0

    def header(self,path):
        import numpy as np
        path=Path(path)
        if path not in self.headers:
            with path.open('rb') as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError('Explicit numeric NPY version required')
                offset=f.tell()
            if fortran or dtype.hasobject or dtype.kind not in 'biuf':raise ValueError('Native C-order numeric cache required')
            size=path.stat().st_size
            if size!=offset+math.prod(shape)*dtype.itemsize:raise ValueError('Complete original NPY byte extent required')
            self.headers[path]=dict(shape=shape,dtype=dtype,offset=offset,size=size)
        return self.headers[path]

    def _block(self,path,start):
        import numpy as np
        key=(path,start)
        if key in self.maps:
            mapping=self.maps.pop(key);self.maps[key]=mapping;return mapping
        size=min(self.block_bytes,self.header(path)['size']-start)
        while self.maps and self.mapped_bytes+size>self.cache_bytes:
            _,old=self.maps.popitem(last=False);self.mapped_bytes-=old.size;old._mmap.close()
        mapping=np.memmap(path,mode='r',dtype=np.uint8,offset=start,shape=(size,))
        self.maps[key]=mapping;self.mapped_bytes+=size;self.peak_mapped_bytes=max(self.peak_mapped_bytes,self.mapped_bytes)
        return mapping

    def flat(self,path,start,count):
        import numpy as np
        path=Path(path);h=self.header(path)
        if isinstance(start,bool) or isinstance(count,bool) or not 0<=start<=start+count<=math.prod(h['shape']):
            raise IndexError('Complete source numeric range required')
        result=np.empty(count,dtype=h['dtype']);first=h['offset']+start*h['dtype'].itemsize;done=0
        while done<count:
            byte=first+done*h['dtype'].itemsize;block_start=byte//self.block_bytes*self.block_bytes
            mapping=self._block(path,block_start);offset=byte-block_start
            available=min(count-done,(len(mapping)-offset)//h['dtype'].itemsize)
            if not available:raise ValueError('Mapping block must preserve numeric item alignment')
            view=np.frombuffer(mapping,dtype=h['dtype'],count=available,offset=offset)
            result[done:done+available]=view;del view;done+=available
        return result

    def close(self):
        for value in self.maps.values():value._mmap.close()
        self.maps.clear();self.mapped_bytes=0


def audit_original_anomaly_targets(source,workspace,output):
    """Offline one-source raw observation binding; never do this per training item."""
    import numpy as np
    import pandas as pd
    directory=Path(source['directory']);proof_path=directory.parent/'pretraining_preparation.json'
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    if source['family']!='anomaly_original' or sha(proof_path)!=source['source_proof_sha256'] or not proof['passed']:
        raise ValueError('Exact independently closed original anomaly source binding required')
    job=proof['job'];raw=job['train_raw'] if job['train_raw'] else job['raw'];raw_path=Path(workspace)/raw['path']
    if sha(raw_path)!=raw['sha256']:raise ValueError('Original raw signal changed')
    frame=pd.read_csv(raw_path,usecols=[0]);observed=np.isfinite(frame.iloc[:,0].to_numpy())
    identifiers=[]
    with raw_path.open('r',encoding='utf-8',newline='') as stream:
        for index,row in enumerate(csv.reader(stream)):
            if row:identifiers.append(index)
    ids=np.asarray(identifiers[1:],dtype=np.int64)
    if len(ids)!=len(observed):raise ValueError('Original header/blank-row identity must agree')
    row_file=directory/'raw_CSV_record_indices.npy';value_file=directory/'values_T.npy'
    record=proof['splits']['train']
    if sha(row_file)!=record['files']['rows']['sha256'] or sha(value_file)!=record['files']['values']['sha256']:
        raise ValueError('Original finite vector and all source row indices must match')
    rows=np.load(row_file,allow_pickle=False);positions=np.searchsorted(ids,rows)
    if np.any(positions>=len(ids)) or not np.array_equal(ids[positions],rows):raise ValueError('Every cache row must map to original raw signal')
    values=np.load(value_file,allow_pickle=False)
    if values.shape!=rows.shape or not np.isfinite(values).all():raise ValueError('Original closed finite model input required')
    targets=observed[positions]
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(kind='verified_all_true' if targets.all() else 'verified_boolean_array',
        source_proof_sha256=source['source_proof_sha256'],raw_signal_sha256=raw['sha256'],
        raw_observation_verified=True,signal_rows=len(rows),valid_targets=int(targets.sum()),
        original_raw_or_cache_modified=False,anomaly_labels_used=False)
    if not targets.all():
        file=output/'valid_targets_T.npy';np.save(file,targets,allow_pickle=False)
        result.update(path=str(file),sha256=sha(file))
    (output/'target_binding.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


class PileCacheReader:
    def __init__(self,sources,*,epoch_seed,original_anomaly_target_bindings=None,cache_bytes=64*2**20,block_bytes=2**20):
        if not sources or len({s['dataset_key'] for s in sources})!=len(sources) or any(s['split']!='train' for s in sources):
            raise ValueError('Unique configured training source identities required')
        self.sources=sources;self.seed=epoch_seed
        # The binding worker adds verified raw profiles while traversing sources.
        # Preserve an explicitly supplied empty registry so those additions reach
        # this reader; never manufacture validity for an unbound source.
        self.target_bindings={} if original_anomaly_target_bindings is None else original_anomaly_target_bindings
        self.arrays=BoundedArrays(cache_bytes,block_bytes);self.groups={}

    def _channels(self,index):
        if index not in self.groups:
            s=self.sources[index]
            # Keep the original raw dataset identity for seeded remainder choice;
            # namespace prefixes only distinguish source records across families.
            raw_key=s['dataset_key'].split(':',1)[1]
            groups=channel_groups(s['native_channels'],self.seed,raw_key,'appendix_D_partition8_remainder_replacement')
            if len(groups)!=s['channel_groups'] or any(len(g)!=s['width'] for g in groups):raise ValueError('All original channel groups and width required')
            self.groups[index]=groups
        return self.groups[index]

    def item(self,source_index,item_index):
        import numpy as np
        if isinstance(source_index,bool) or not 0<=source_index<len(self.sources):raise IndexError('Configured source identity required')
        s=self.sources[source_index]
        if isinstance(item_index,bool) or not 0<=item_index<s['items']:raise IndexError('Complete source item identity required')
        base,group=divmod(item_index,s['channel_groups']);channels=self._channels(source_index)[group]
        if s['items']!=s['training_base_cases_or_windows']*s['channel_groups']:raise ValueError('No implicit item truncation or duplication allowed')
        directory=Path(s['directory'])/('TRAIN' if s['family']=='classification' else '')
        provenance=dict(dataset_key=s['dataset_key'],family=s['family'],source_index=source_index,
            item_index=item_index,base_case_or_window=base,group_index=group,channel_indices=channels)
        if s['family'] in ['classification','monash']:
            value_file=directory/'values.npy';shape=self.arrays.header(value_file)['shape']
            if len(shape)!=3 or shape[1:]!=(s['native_channels'],256):raise ValueError('All original case/channel/time cache axes required')
            if s['family']=='classification':
                case=base;row_file=directory/'original_instance_rows.npy'
                provenance['original_instance_row']=int(self.arrays.flat(row_file,case,1)[0])
            else:
                start,end=s['source_train_shuffle_positions']
                if base>=end-start:raise ValueError('Source training shuffle position required')
                case=int(self.arrays.flat(directory/'source_shuffle_order.npy',start+base,1)[0])
                provenance['original_series_row']=int(self.arrays.flat(directory/'source_series_rows.npy',case,1)[0])
            if not 0<=case<shape[0]:raise ValueError('Exact original case row required')
            values=np.stack([self.arrays.flat(value_file,(case*shape[1]+c)*256,256) for c in channels])
            padding=self.arrays.flat(directory/'input_mask.npy',case*256,256)
            valid=np.broadcast_to(padding.astype(bool),(len(channels),256)).copy()
            provenance['target_validity_rule']='source_cached_padding_mask;not_independent_raw_missingness_certification'
        elif s['family']=='long_forecast':
            value_file=directory/'values_TC.npy';shape=self.arrays.header(value_file)['shape']
            if len(shape)!=2 or shape[1]!=s['native_channels'] or base+256>shape[0]:raise ValueError('Complete original TC time window required')
            matrix=self.arrays.flat(value_file,base*shape[1],256*shape[1]).reshape(256,shape[1])
            values=np.array(matrix[:,list(channels)].T,copy=True,order='C')
            padding=self.arrays.flat(directory/'input_mask_window.npy',0,256)
            valid=np.broadcast_to(padding.astype(bool),values.shape).copy()
            provenance['raw_CSV_rows']=self.arrays.flat(directory/'original_CSV_rows.npy',base,256)
            provenance['target_validity_rule']='source_transformed_finite_inputs_and_padding;fullCSV_interpolation_retained'
        elif s['family'] in ['anomaly_original','anomaly_finite_control']:
            value_file=directory/'values_T.npy';shape=self.arrays.header(value_file)['shape']
            if len(shape)!=1 or s['native_channels']!=1 or base+256>shape[0]:raise ValueError('Complete univariate source time window required')
            values=self.arrays.flat(value_file,base,256).reshape(1,256)
            padding=self.arrays.flat(directory/'input_mask_window.npy',0,256)
            provenance['raw_CSV_rows']=self.arrays.flat(directory/'raw_CSV_record_indices.npy',base,256)
            if s['family']=='anomaly_finite_control':
                valid=self.arrays.flat(directory/'valid_targets_T.npy',base,256).reshape(1,256)
                provenance['residual_filled']=self.arrays.flat(directory/'residual_filled_T.npy',base,256)
            else:
                binding=self.target_bindings.get(s['dataset_key'])
                if (not binding or not binding['raw_observation_verified']
                        or binding['source_proof_sha256']!=s['source_proof_sha256'] or binding['signal_rows']!=shape[0]):
                    raise ValueError('Original anomaly raw observation target binding required;no silent all-true fallback')
                if binding['kind']=='verified_all_true':valid=np.ones((1,256),dtype=bool)
                elif binding['kind']=='verified_boolean_array':valid=self.arrays.flat(binding['path'],base,256).reshape(1,256)
                else:raise ValueError('Exact original target validity kind required')
            provenance['target_validity_rule']='raw_observed_AND_source_finite;mandatory_in_reconstruction_loss'
        else:raise ValueError('Explicit registered cache family required')
        if (values.shape!=(s['width'],256) or not np.isfinite(values).all() or padding.shape!=(256,)
                or not np.isin(padding,[0,1]).all() or valid.shape!=values.shape or valid.dtype!=np.dtype(bool)):
            raise ValueError('Complete finite values, source padding and bool target identity required')
        valid=valid & padding.astype(bool)[None,:]
        return dict(values=values,input_mask=padding,valid_targets=valid,provenance=provenance)

    def batch(self,reference):
        import numpy as np
        examples=[self.item(source,index) for source,index in reference.items]
        if not examples or any(e['values'].shape!=(reference.width,256) for e in examples):raise ValueError('Exact homogeneous width batch required')
        return dict(values=np.stack([e['values'] for e in examples]),input_mask=np.stack([e['input_mask'] for e in examples]),
            valid_targets=np.stack([e['valid_targets'] for e in examples]),provenance=[e['provenance'] for e in examples],
            examples_with_no_valid_target=sum(not e['valid_targets'].any() for e in examples))

    def close(self):self.arrays.close()
