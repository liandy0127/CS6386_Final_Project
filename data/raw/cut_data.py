import ijson

def extract_first_n_items(input_file, output_file, n=1000):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write('{"items": [\n')
        for i, line in enumerate(infile):
            if i >= n:
                break
            line = line.rstrip(',\n') + '\n'
            outfile.write(line)
        outfile.write(']}\n')

extract_first_n_items('Video_Games.jsonl', 'Video_Games_subset.jsonl')
extract_first_n_items('meta_Video_Games.jsonl', 'meta_Video_Games_subset.jsonl')
