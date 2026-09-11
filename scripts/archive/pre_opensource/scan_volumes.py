#!/usr/bin/env python3
"""Scan complete tar.gz volumes to identify which datasets they contain.

This helps determine which volumes need to be extracted for seg/structure tasks.
"""

import gzip
import tarfile
import json
import sys
from pathlib import Path
from collections import defaultdict

VOLUME_ROOT = Path('/media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M')
COMPLETE_VOLUMES = [
    '000', '003', '004', '005', '006', '007', '008', '009', '010', '011',
    '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
    '022', '023', '024', '025', '026', '027', '028', '029', '030', '031',
    '032', '033', '034', '035', '036', '038', '039', '040', '041', '042',
    '043', '050', '051', '052', '058', '059', '060', '061', '062', '068',
    '069', '070', '081', '083', '084', '085', '090', '091', '092', '132',
    '134', '135', '149', '150', '151', '152', '153', '154', '155', '156',
    '157', '160', '162', '163', '164', '165', '166', '167', '168', '169',
    '170', '171', '172', '173', '174', '175', '176', '177', '178', '179',
    '180', '181', '182', '183', '184', '185', '186', '300', '301', '305',
    '342', '343', '344'
]

# Datasets we need for seg/structure tasks
NEEDED_DATASETS = {
    'coco2017', 'coco2014',  # COCO for seg
    'Cityscapes',  # Cityscapes for seg
    'KITTI',  # KITTI for seg
    'LoveDA',  # LoveDA for seg
    'hypersim',  # Hypersim for seg
    'wild_rgbd', 'WildRGBD', 'backpack',  # WildRGBD for structure
    'streethazards', 'StreetHazards',  # StreetHazards for seg
    'tas500', 'TAS500',  # TAS500 for seg
    'trashcan', 'TrashCan',  # TrashCan for seg
    'ttpla', 'TTPLA',  # TTPLA for seg
    'zerowaste', 'ZeroWaste',  # ZeroWaste for seg
    'vipseg', 'ViPSeg',  # ViPSeg for seg
    'entityv2', 'EntityV2',  # EntityV2 for seg
    'flickr', 'flickr30k',  # Flickr for seg
    'PSG',  # PSG for seg
    'refcocog', 'RefCOCOg',  # RefCOCOg for seg
}

def scan_volume(vol_id: str, max_entries: int = 100000) -> dict:
    """Scan a volume and return statistics about its contents."""
    vol_path = VOLUME_ROOT / f'SenseNova-Vision-Corpus-50M.tar.gz{vol_id}'
    
    if not vol_path.exists():
        return {'error': 'not found'}
    
    stats = {
        'datasets': defaultdict(int),
        'total_entries': 0,
        'scanned_entries': 0,
    }
    
    try:
        with open(vol_path, 'rb') as f:
            gz = gzip.GzipFile(fileobj=f)
            tar = tarfile.open(fileobj=gz, mode='r|')
            
            for i, member in enumerate(tar):
                if i >= max_entries:
                    break
                
                stats['scanned_entries'] = i + 1
                parts = member.name.split('/')
                if len(parts) > 1:
                    dataset = parts[1]
                    stats['datasets'][dataset] += 1
                    
    except Exception as e:
        stats['error'] = str(e)
    
    stats['datasets'] = dict(stats['datasets'])
    return stats


def main():
    results = {}
    
    print("Scanning complete volumes for dataset content...")
    print(f"Total volumes to scan: {len(COMPLETE_VOLUMES)}")
    
    for i, vol_id in enumerate(COMPLETE_VOLUMES):
        print(f"[{i+1}/{len(COMPLETE_VOLUMES)}] Scanning volume {vol_id}...", end=' ', flush=True)
        stats = scan_volume(vol_id)
        results[vol_id] = stats
        
        if 'error' in stats:
            print(f"ERROR: {stats['error']}")
        else:
            datasets = list(stats['datasets'].keys())
            print(f"Found {len(datasets)} datasets: {datasets[:5]}{'...' if len(datasets) > 5 else ''}")
    
    # Save results
    output_file = Path('/media/linxu/thinkHub/Thinking-face/open-vision-banana/output/volume_scan_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Summary
    print("\n=== Summary ===")
    dataset_volumes = defaultdict(list)
    for vol_id, stats in results.items():
        if 'error' not in stats:
            for dataset in stats['datasets']:
                dataset_volumes[dataset].append(vol_id)
    
    print("\nDataset locations:")
    for dataset in sorted(dataset_volumes.keys()):
        vols = dataset_volumes[dataset]
        needed = "★" if dataset in NEEDED_DATASETS else " "
        print(f"  {needed} {dataset}: volumes {vols[:10]}{'...' if len(vols) > 10 else ''} ({len(vols)} total)")


if __name__ == '__main__':
    main()
