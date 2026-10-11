"""Read every original derived test mask/category; no model or score generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.flow_matching.flow_mismatching_full_training_contract_v1 import ROOT, read, sha, verify_receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queue = read(ROOT / args.queue); verify_receipts(queue['source_receipts'])
    import numpy as np
    from scripts.flow_matching.flow_mismatching_image_metrics_v1 import regions_per_image
    records, seen = [], set()
    for case in queue['cases']:
        if case['dataset'] in seen:
            continue
        seen.add(case['dataset'])
        training = read(ROOT / case['training_configuration'])
        dataset = read(ROOT / training['dataset_configuration'])
        verify_receipts(dataset['source_receipts']); verify_receipts(dataset['array_receipts'])
        derived_path = next(x['path'] for x in dataset['source_receipts']
                            if Path(x['path']).name == 'full_derived_image_preprocessing.json')
        names = read(ROOT / derived_path)['original_categories']
        test = {x['field']: np.load(x['path'], mmap_mode='r', allow_pickle=False)
                for x in dataset['array_receipts'] if x['split'] == 'test' and x['field'] != 'images'}
        labels, masks, categories = test['labels'], test['masks'], test['categories']
        n = dataset['expected_native_splits']['test']
        if masks.shape != (n, 128, 128) or labels.shape != (n,) or categories.shape != (n,):
            raise ValueError('Complete original native test-array geometry required')
        rows = []
        for category, name in enumerate(names):
            subset = categories == category
            selected_masks, selected_labels = masks[subset], labels[subset]
            positives = int(selected_masks.sum()); negatives = int(selected_masks.size - positives)
            positive_images = int(selected_labels.sum()); negative_images = int(len(selected_labels) - positive_images)
            if min(positives, negatives, positive_images, negative_images) <= 0:
                raise ValueError('Original category lacks required positive/background/image units: ' + name)
            components = regions_per_image(selected_masks, queue['metrics']['region_connectivity'])
            rows.append(dict(category=name, category_index=category, full_test_images=int(subset.sum()),
                original_anomalous_image_labels=positive_images, original_normal_image_labels=negative_images,
                complete_resized128_anomalous_pixels=positives, complete_resized128_background_pixels=negatives,
                complete_resized128_connected_regions=sum(count for _, count in components),
                original_anomalous_images_with_empty_resized128_mask=int(np.sum((selected_labels == 1) & (selected_masks.sum((1, 2)) == 0)))))
        if sum(r['full_test_images'] for r in rows) != n or len(rows) != dataset['expected_original_category_count']:
            raise ValueError('Every native category and original test image must be included')
        records.append(dict(dataset=case['original_dataset_name'], complete_original_test_images=n,
            complete_original_categories=len(names), per_category=rows,
            all_original_source_and_eight_array_hashes_verified=True,
            dataset_configuration=training['dataset_configuration'], dataset_sha256=sha(ROOT / training['dataset_configuration'])))
    if seen != {'mvtec', 'visa'}:
        raise ValueError('Both complete original image datasets required')
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(passed=True, captured_utc=datetime.now(timezone.utc).isoformat(),
            queue_configuration=args.queue, queue_sha256=sha(ROOT / args.queue), records=records,
            complete_original_test_images=sum(r['complete_original_test_images'] for r in records),
            complete_original_categories=sum(r['complete_original_categories'] for r in records),
            auroc_AP_F1_AUPRO_defined_for_every_native_category=True,
            region_connectivity=queue['metrics']['region_connectivity'],
            original_full_test_images_subsampled=0, original_files_written_or_overwritten=0,
            trained_model_forward_calls=0, native_benchmark_performance=False,
            boundary='Complete ground-truth scoring-readiness audit only; no model score or performance exists.'), indent=2) + '\n')
    print(dict(full_original_test_images=sum(r['complete_original_test_images'] for r in records),
               all_original_categories=sum(r['complete_original_categories'] for r in records), passed=True))


if __name__ == '__main__':
    main()
