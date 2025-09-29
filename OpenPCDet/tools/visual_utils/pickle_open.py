import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument("pkl_path", type=str, help="Path to .pkl file")
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as f:
        infos = pickle.load(f)
    
    print("총 샘플 개수:", len(infos))
    print("첫 번째 항목 타입:", type(infos[0]))
    print("첫 번째 항목 키들:", infos[0].keys())

    # point_cloud 키가 있으면 그 안을 출력
    if "point_cloud" in infos[0]:
        print("\npoint_cloud 내용:")
        for k, v in infos[0]["point_cloud"].items():
            print(f"  {k}: {v}")

    # annos 키가 있으면 일부 출력
    if "annos" in infos[0]:
        print("\nannos 키들:", infos[0]["annos"].keys())

if __name__ == "__main__" :
    main()