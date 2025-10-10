from pathlib import Path

DATAPATH = Path("./data/custom_av_hybrid/points")

def main() :
    count = sum(1 for _ in DATAPATH.glob("*.npy"))
    print(f"총 {count}개의 .npy 파일이 있습니다.")

if __name__ == "__main__":
    main()