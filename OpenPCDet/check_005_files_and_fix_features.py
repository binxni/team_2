#!/usr/bin/env python3
"""
8자리 frame_id 중 005로 시작하는 .npy 파일들의 feature를 확인하고 ['x', 'y', 'z', 'intensity']만 남기도록 수정하는 스크립트

이 스크립트는:
1. 지정된 경로에서 8자리 frame_id 중 005로 시작하는 .npy 파일들을 찾습니다
2. 각 파일의 shape과 feature 정보를 확인합니다  
3. ['x', 'y', 'z', 'intensity'] 이외의 feature가 있으면 이를 제거합니다
4. 수정된 파일을 원본 위치에 저장합니다

frame_id 형식:
- 001XXXXX, 002XXXXX, 003XXXXX, 005XXXXX, 000XXXXX 등의 8자리 중에서
- 005로 시작하는 파일만 대상으로 합니다

사용법:
    python check_005_files_and_fix_features.py

주의사항:
    - 원본 파일이 덮어씌워지므로 백업을 먼저 만드는 것을 권장합니다
    - numpy 배열의 마지막 차원이 feature 차원이라고 가정합니다
"""

import os
import numpy as np
import glob
from pathlib import Path
import shutil
from datetime import datetime


def check_sample_file_structure(points_dir):
    """샘플 파일의 구조를 확인합니다."""
    print("=== 샘플 파일 구조 확인 ===")
    
    # 첫 번째 파일 몇 개를 확인
    sample_files = glob.glob(os.path.join(points_dir, "*.npy"))[:5]
    
    if not sample_files:
        print("경고: .npy 파일을 찾을 수 없습니다.")
        return None
        
    for file_path in sample_files:
        try:
            data = np.load(file_path)
            print(f"파일: {os.path.basename(file_path)}")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            if len(data.shape) == 2:
                print(f"  Features (차원): {data.shape[1]}")
                print(f"  Points (포인트 수): {data.shape[0]}")
                if data.shape[1] >= 4:
                    print(f"  첫 번째 포인트 예시: {data[0, :min(6, data.shape[1])]}")
            print()
        except Exception as e:
            print(f"파일 {file_path} 로드 중 오류: {e}")
    
    return sample_files[0] if sample_files else None


def find_005_files(points_dir):
    """8자리 frame_id 중에서 005로 시작하는 .npy 파일들을 찾습니다."""
    print("=== 8자리 frame_id 중 005로 시작하는 파일 검색 ===")
    
    # 모든 .npy 파일을 가져온 후 필터링
    all_files = glob.glob(os.path.join(points_dir, "*.npy"))
    found_files = []
    
    for file_path in all_files:
        basename = os.path.basename(file_path)
        frame_id = basename.replace('.npy', '')
        
        # 8자리이고 005로 시작하는 파일만 선택
        if len(frame_id) == 8 and frame_id.startswith('050'):
            found_files.append(file_path)
    
    found_files.sort()
    
    print(f"찾은 8자리 005 파일 수: {len(found_files)}")
    
    if found_files:
        print("처음 10개 파일:")
        for i, file_path in enumerate(found_files[:10]):
            print(f"  {i+1}. {os.path.basename(file_path)}")
        if len(found_files) > 10:
            print(f"  ... 그리고 {len(found_files) - 10}개 더")
    else:
        print("8자리 005로 시작하는 파일을 찾을 수 없습니다.")
        print("전체 파일 중 일부를 확인해보겠습니다...")
        
        # 전체 파일에서 파일명 패턴 확인
        if all_files:
            print(f"전체 파일 수: {len(all_files)}")
            print("파일명 예시 (처음 20개):")
            for i, file_path in enumerate(all_files[:20]):
                basename = os.path.basename(file_path)
                frame_id = basename.replace('.npy', '')
                print(f"  {basename} (길이: {len(frame_id)})")
            
            # 005로 시작하는 파일들 (길이 무관) 검색
            files_with_005 = []
            eight_digit_files = []
            
            for file_path in all_files:
                basename = os.path.basename(file_path)
                frame_id = basename.replace('.npy', '')
                
                if frame_id.startswith('050'):
                    files_with_005.append(file_path)
                
                if len(frame_id) == 8:
                    eight_digit_files.append(file_path)
            
            if files_with_005:
                print(f"\n005로 시작하는 파일 수 (길이 무관): {len(files_with_005)}")
                for file_path in files_with_005[:10]:
                    basename = os.path.basename(file_path)
                    frame_id = basename.replace('.npy', '')
                    print(f"  {basename} (길이: {len(frame_id)})")
            
            if eight_digit_files:
                print(f"\n8자리 파일 수: {len(eight_digit_files)}")
                # 8자리 파일들의 접두사 확인
                prefixes = {}
                for file_path in eight_digit_files[:50]:  # 처음 50개만 확인
                    basename = os.path.basename(file_path)
                    frame_id = basename.replace('.npy', '')
                    prefix = frame_id[:3]
                    prefixes[prefix] = prefixes.get(prefix, 0) + 1
                
                print("8자리 파일들의 접두사 분포 (처음 50개 기준):")
                for prefix, count in sorted(prefixes.items()):
                    print(f"  {prefix}: {count}개")
    
    return found_files


def check_point_cloud_features(file_path):
    """포인트 클라우드 파일의 feature를 확인합니다."""
    try:
        data = np.load(file_path)
        
        if len(data.shape) != 2:
            return None, f"예상과 다른 shape: {data.shape} (2D 배열이어야 함)"
        
        num_points, num_features = data.shape
        
        return {
            'shape': data.shape,
            'num_points': num_points,
            'num_features': num_features,
            'dtype': data.dtype,
            'sample_data': data[:3] if num_points >= 3 else data  # 처음 3개 포인트
        }, None
        
    except Exception as e:
        return None, f"파일 로드 오류: {e}"


def fix_point_cloud_features(file_path, backup_dir=None):
    """포인트 클라우드에서 ['x', 'y', 'z', 'intensity']만 남기고 나머지 제거합니다."""
    try:
        # 원본 데이터 로드
        data = np.load(file_path)
        
        if len(data.shape) != 2:
            return False, f"예상과 다른 shape: {data.shape}"
        
        num_points, num_features = data.shape
        
        # 4개보다 적거나 같으면 수정할 필요 없음
        if num_features <= 4:
            return True, f"이미 4개 이하의 feature ({num_features}개)를 가지고 있음"
        
        # 백업 생성 (선택사항)
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
        
        # 처음 4개 feature만 유지 (x, y, z, intensity)
        fixed_data = data[:, :4].copy()
        
        # 수정된 데이터 저장
        np.save(file_path, fixed_data)
        
        return True, f"Feature 수정 완료: {num_features} -> 4"
        
    except Exception as e:
        return False, f"수정 중 오류: {e}"


def main():
    """메인 함수"""
    # 설정
    points_dir = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_range64/waymo_npy/points"
    backup_dir = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_range64/waymo_npy/points"
    
    print("포인트 클라우드 Feature 검사 및 수정 도구")
    print("=" * 50)
    print(f"대상 디렉토리: {points_dir}")
    print("대상 파일: 8자리 frame_id 중 005로 시작하는 .npy 파일")
    print()
    
    # 디렉토리 존재 확인
    if not os.path.exists(points_dir):
        print(f"오류: 디렉토리가 존재하지 않습니다: {points_dir}")
        return
    
    # 1. 샘플 파일 구조 확인
    sample_file = check_sample_file_structure(points_dir)
    if not sample_file:
        print("샘플 파일을 찾을 수 없어 종료합니다.")
        return
    
    # 2. 005로 시작하는 파일 찾기
    target_files = find_005_files(points_dir)
    if not target_files:
        print("처리할 파일을 찾을 수 없어 종료합니다.")
        return
    
    print(f"\n처리할 파일 수: {len(target_files)}")
    
    # 3. 사용자 확인
    proceed = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
    if proceed != 'y':
        print("작업이 취소되었습니다.")
        return
    
    # 4. 백업 디렉토리 생성 여부 확인
    create_backup = input("백업을 생성하시겠습니까? (Y/n): ").strip().lower()
    if create_backup != 'n':
        print(f"백업 디렉토리: {backup_dir}")
    else:
        backup_dir = None
    
    # 5. 파일 처리
    print("\n=== 파일 처리 시작 ===")
    
    processed_count = 0
    modified_count = 0
    error_count = 0
    
    for i, file_path in enumerate(target_files):
        print(f"[{i+1}/{len(target_files)}] {os.path.basename(file_path)}")
        
        # feature 확인
        info, error = check_point_cloud_features(file_path)
        if error:
            print(f"  오류: {error}")
            error_count += 1
            continue
        
        print(f"  Shape: {info['shape']}, Features: {info['num_features']}")
        
        # feature 수정 필요한지 확인
        if info['num_features'] > 4:
            success, message = fix_point_cloud_features(file_path, backup_dir)
            if success:
                print(f"  ✓ {message}")
                modified_count += 1
            else:
                print(f"  ✗ {message}")
                error_count += 1
        else:
            print(f"  - 수정 불필요 (이미 {info['num_features']}개 feature)")
        
        processed_count += 1
        
        # 진행상황 표시
        if (i + 1) % 100 == 0:
            print(f"\n진행상황: {i+1}/{len(target_files)} 완료\n")
    
    # 6. 결과 요약
    print("\n=== 처리 결과 ===")
    print(f"총 처리된 파일: {processed_count}")
    print(f"수정된 파일: {modified_count}")
    print(f"오류 발생: {error_count}")
    
    if backup_dir and os.path.exists(backup_dir):
        backup_files = len(os.listdir(backup_dir))
        print(f"백업된 파일: {backup_files}")
        print(f"백업 위치: {backup_dir}")
    
    print("\n작업 완료!")


if __name__ == "__main__":
    main()