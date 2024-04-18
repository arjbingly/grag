import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fail', action='store_true')
    args = parser.parse_args()
    readme_path = 'README.md'  # Path to your README file

    with open(readme_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'img.shields.io' in line and ('passing' in line or 'failing' in line):
            if args.fail:
                lines[i] = line.replace('passing', 'failing').replace('darggreen', 'red')
            else:
                lines[i] = line.replace('failing', 'passing').replace('red', 'darggreen')
            print('README file has been updated.')

    with open(readme_path, 'w') as file:
        file.writelines(lines)
