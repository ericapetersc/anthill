import argparse
import errno
import os

import numpy as np
import pandas as pd

from ant_colony_optimization import AntColonyOptimization, Datum

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="data filename")
parser.add_argument(
    "-t", "--header", help="header exits in file", type=bool, default=True
)
parser.add_argument("-l", "--label", help="label column", type=int, default=-1)
parser.add_argument("-r", "--rows", help="number of grid rows", type=int, default=60)
parser.add_argument(
    "-c", "--columns", help="number of grid columns", type=int, default=60
)
parser.add_argument("-n", "--ants", help="number of ants", type=int, default=20)
parser.add_argument("-v", "--ratio", help="ants vision ratio", type=int, default=2)
parser.add_argument("-p", "--kp", help="pick func constant", type=float, default=0.1)
parser.add_argument("-d", "--kd", help="drop func constant", type=float, default=0.15)
parser.add_argument(
    "-a", "--alpha", help="dissimilarity scale", type=float, default=1.5
)
parser.add_argument(
    "-i", "--iterations", help="number of iterations", type=int, default=150000
)
args = parser.parse_args()


def input_data():
    if not os.path.isfile(args.filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.filename)

    header = 0 if args.header else None
    data = pd.read_csv(args.filename, header=header)
    labels = np.empty(data.shape[0], dtype=object)
    if args.label != -1:
        labels, _ = pd.factorize(data.iloc[:, args.label].to_list())
        data = data.drop(data.columns[[args.label]], axis=1)
    return data.to_numpy(), labels


def create_objects(data, labels):
    objs = []
    for narray, label in zip(data, labels):
        objs.append(Datum(narray, label))
    return objs


def main():
    data, labels = input_data()
    aco = AntColonyOptimization(
        **{
            "data": create_objects(data, labels),
            "rows": args.rows,
            "columns": args.columns,
            "ants": args.ants,
            "ratio": args.ratio,
            "kp": args.kp,
            "kd": args.kd,
            "alpha": args.alpha,
            "iterations": args.iterations,
        }
    )
    aco.execute()


if __name__ == "__main__":
    main()
