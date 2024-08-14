import json

if __name__ == '__main__':
    file = "boolq.jsonl"
    with open(file, "r") as f:
        datas = f.readlines()
        datas = [json.loads(data) for data in datas]

    with open("../boolq.json", "w")as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)