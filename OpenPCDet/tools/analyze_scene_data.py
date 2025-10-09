#!/usr/bin/env python3
"""
Scene별 데이터 개수 분석 스크립트
Point cloud data의 각 scene별 train/val 개수를 분석합니다.
"""

import os
from pathlib import Path
from collections import defaultdict

def analyze_scene_data():
    """각 scene별 64channel과 128channel의 train과 val 개수를 분석"""
    
    # 데이터 경로 설정
    data_dir = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_64/ImageSets")
    
    # 기본 데이터셋 파일들
    base_files = ["train.txt", "val.txt"]
    
    # 채널별 파일들
    channel_files = [
        "train_64.txt",
        "train_128.txt", 
        "val_64.txt",
        "val_128.txt"
    ]
    
    # Scene 매핑
    scene_mapping = {
        "000": "Scene 1",
        "001": "Scene 2", 
        "002": "Scene 3",
        "003": "Scene 4"
    }
    
    # 결과 저장용 딕셔너리
    results = {}
    
    print("=" * 80)
    print("Point Cloud Data Scene별 개수 분석")
    print("=" * 80)
    
    # 1단계: 각 채널별 파일에서 데이터 읽기
    channel_data = {}
    for file_name in channel_files:
        file_path = data_dir / file_name
        
        if not file_path.exists():
            print(f"⚠️  파일을 찾을 수 없습니다: {file_path}")
            continue
            
        channel_data[file_name] = set()
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        channel_data[file_name].add(line)
        except Exception as e:
            print(f"❌ {file_name} 읽기 실패: {e}")
    
    # 2단계: train.txt와 val.txt를 기준으로 scene별 분석
    for base_file in base_files:
        file_path = data_dir / base_file
        
        if not file_path.exists():
            print(f"⚠️  파일을 찾을 수 없습니다: {file_path}")
            continue
            
        # Scene별 카운트 (64채널, 128채널별로)
        scene_64_counts = defaultdict(int)
        scene_128_counts = defaultdict(int)
        total_64_count = 0
        total_128_count = 0
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 첫 3자리로 scene 판별
                        scene_prefix = line[:3]
                        if scene_prefix in scene_mapping:
                            # 64채널 체크
                            if base_file == "train.txt":
                                if line in channel_data.get("train_64.txt", set()):
                                    scene_64_counts[scene_prefix] += 1
                                    total_64_count += 1
                                if line in channel_data.get("train_128.txt", set()):
                                    scene_128_counts[scene_prefix] += 1
                                    total_128_count += 1
                            else:  # val.txt
                                if line in channel_data.get("val_64.txt", set()):
                                    scene_64_counts[scene_prefix] += 1
                                    total_64_count += 1
                                if line in channel_data.get("val_128.txt", set()):
                                    scene_128_counts[scene_prefix] += 1
                                    total_128_count += 1
                        else:
                            print(f"⚠️  알 수 없는 scene prefix: {scene_prefix} in {base_file}")
            
            # 결과 저장
            results[base_file] = {
                'scene_64_counts': dict(scene_64_counts),
                'scene_128_counts': dict(scene_128_counts),
                'total_64_count': total_64_count,
                'total_128_count': total_128_count
            }
            
            # 결과 출력
            print(f"\n📁 {base_file}")
            print("-" * 50)
            print("  🔵 64채널:")
            for scene_code in sorted(scene_64_counts.keys()):
                scene_name = scene_mapping[scene_code]
                count = scene_64_counts[scene_code]
                percentage = (count / total_64_count * 100) if total_64_count > 0 else 0
                print(f"    {scene_name} ({scene_code}): {count:,}개 ({percentage:.1f}%)")
            print(f"    📊 64채널 총 개수: {total_64_count:,}개")
            
            print("  🟡 128채널:")
            for scene_code in sorted(scene_128_counts.keys()):
                scene_name = scene_mapping[scene_code]
                count = scene_128_counts[scene_code]
                percentage = (count / total_128_count * 100) if total_128_count > 0 else 0
                print(f"    {scene_name} ({scene_code}): {count:,}개 ({percentage:.1f}%)")
            print(f"    📊 128채널 총 개수: {total_128_count:,}개")
            
        except Exception as e:
            print(f"❌ {base_file} 읽기 실패: {e}")
    
    # 요약 테이블 출력
    print("\n" + "=" * 80)
    print("📊 요약 테이블")
    print("=" * 80)
    
    # 헤더 출력
    print(f"{'Scene':<12} {'train_64':<12} {'train_128':<12} {'val_64':<12} {'val_128':<12} {'Total':<12}")
    print("-" * 80)
    
    # Scene별 합계 계산
    scene_totals = defaultdict(int)
    
    for scene_code, scene_name in scene_mapping.items():
        counts = []
        scene_total = 0
        
        for file_name in files_to_analyze:
            if file_name in results:
                count = results[file_name]['scene_counts'].get(scene_code, 0)
                counts.append(f"{count:,}")
                scene_total += count
            else:
                counts.append("N/A")
        
        scene_totals[scene_name] = scene_total
        print(f"{scene_name:<12} {counts[0]:<12} {counts[1]:<12} {counts[2]:<12} {counts[3]:<12} {scene_total:,}")
    
    # 전체 합계
    print("-" * 80)
    total_counts = []
    grand_total = 0
    for file_name in files_to_analyze:
        if file_name in results:
            total = results[file_name]['total_count']
            total_counts.append(f"{total:,}")
            grand_total += total
        else:
            total_counts.append("N/A")
    
    print(f"{'Total':<12} {total_counts[0]:<12} {total_counts[1]:<12} {total_counts[2]:<12} {total_counts[3]:<12} {grand_total:,}")
    
    # 채널별/데이터셋별 분석
    print("\n" + "=" * 80)
    print("📈 채널별/데이터셋별 분석")
    print("=" * 80)
    
    # Train vs Val 비교 (64채널)
    if "train_64.txt" in results and "val_64.txt" in results:
        train_64_total = results["train_64.txt"]["total_count"]
        val_64_total = results["val_64.txt"]["total_count"]
        total_64 = train_64_total + val_64_total
        
        print(f"\n🔵 64채널 데이터:")
        print(f"  Train: {train_64_total:,}개 ({train_64_total/total_64*100:.1f}%)")
        print(f"  Val:   {val_64_total:,}개 ({val_64_total/total_64*100:.1f}%)")
        print(f"  Total: {total_64:,}개")
    
    # Train vs Val 비교 (128채널)
    if "train_128.txt" in results and "val_128.txt" in results:
        train_128_total = results["train_128.txt"]["total_count"]
        val_128_total = results["val_128.txt"]["total_count"]
        total_128 = train_128_total + val_128_total
        
        print(f"\n🟡 128채널 데이터:")
        print(f"  Train: {train_128_total:,}개 ({train_128_total/total_128*100:.1f}%)")
        print(f"  Val:   {val_128_total:,}개 ({val_128_total/total_128*100:.1f}%)")
        print(f"  Total: {total_128:,}개")
    
    # Scene별 비율 분석
    print(f"\n🎯 Scene별 데이터 분포:")
    for scene_name, count in scene_totals.items():
        percentage = (count / grand_total * 100) if grand_total > 0 else 0
        print(f"  {scene_name}: {count:,}개 ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ 분석 완료!")
    print("=" * 80)

if __name__ == "__main__":
    analyze_scene_data()