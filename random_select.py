import argparse
import random


parser = argparse.ArgumentParser(description='Flask server script')
parser.add_argument('--input_path', type=str,
                    default='sentences/KCTTS.txt')
parser.add_argument('--select_num', type=int, default=100, help='number of random selected samples')
args = parser.parse_args()


if __name__ == '__main__':
    input_path = args.input_path
    output_path = input_path.replace('.txt', '_random.txt')
    select_num = args.select_num
    with open(input_path, encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    
    selected = random.sample(sentences, select_num)
    
    wf = open(output_path, 'w', encoding='utf-8')
    for i, s in enumerate(selected):
        if i+1 == select_num:
            wf.write(s)
        else:
            wf.write(s + '\n')
    wf.close()