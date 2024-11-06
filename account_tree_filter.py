import os
import sys

import pandas as pandas


def main(file_path):
    csv = pandas.read_csv(file_path)
    filtered = csv[(csv['Full Account Name'].str.startswith('Assets:')) & (csv['Hidden'] == 'F')]

    new_basename = ".".join(os.path.basename(file_path).split(".")[:-1]) + "_filtered.csv"
    filtered.to_csv(os.path.join(os.path.dirname(file_path), new_basename), index=False)


if __name__ == "__main__":
    args = sys.argv
    main(args[1])
