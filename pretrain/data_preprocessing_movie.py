from collections import namedtuple
import csv
import math
import json
import random

import argparse
from re import template
from torch.utils.data import DataLoader, Dataset
from movie_template import task_subgroup_2 as task_sequential_movie
from movie_template import task_subgroup_5 as task_direct_movie
from movie_template import task_subgroup_1 as task_movie_information


def divide_chunks(k, l, n):
    chunked_list = []
    for i in range(0, len(l), n):
        if len(l[i : i + n]) >= n - 3:
            chunked_list.append([k, l[i : i + n]])
    return chunked_list


def build_movie_dataset(args):
    with open(args.data_dir + "movie/movies.dat", "r", encoding="ISO-8859-1") as f:
        movie_data = f.read()

    movie_data = movie_data.split("\n")
    movie_data.remove("")  # 3883 movies in total, last index in 3952
    movie = {"index": "", "title": "", "category": []}
    all_movies = {}
    for d in movie_data:
        d = d.split("::")
        categories = d[2].split("|")
        movie["index"] = d[0]
        movie["title"] = d[1]
        movie["category"] = categories
        all_movies[d[0]] = movie
        movie = {"index": "", "title": "", "category": []}

    with open(args.data_dir + "movie/users.dat", encoding="ISO-8859-1") as f:
        user_data = f.read()

    user_data = user_data.split("\n")
    user_data.remove("")
    user = {"index": "", "gender": "", "age": "", "occupation": ""}
    all_users = {}
    for d in user_data:
        d = d.split("::")
        user["index"] = d[0]
        user["gender"] = d[1]
        user["age"] = d[2]
        user["occupation"] = d[3]
        all_users[d[0]] = user
        user = {"index": "", "gender": "", "age": "", "occupation": ""}

    with open(args.data_dir + "movie/ratings.dat", encoding="ISO-8859-1") as f:
        rating_data = f.read()

    rating_data = rating_data.split("\n")
    rating_data.remove("")
    rating = {"user": "", "movie": "", "rating": "", "time": ""}
    user_ratings = {}
    for d in rating_data:
        d = d.split("::")
        rating["user"] = d[0]
        rating["movie"] = all_movies[d[1]]
        rating["rating"] = d[2]
        rating["time"] = d[3]
        if d[0] not in user_ratings:
            user_ratings[d[0]] = [rating]
        else:
            user_ratings[d[0]].append(rating)
        rating = {"user": "", "movie": "", "rating": "", "time": ""}

    for k in user_ratings.keys():
        user_ratings[k] = sorted(user_ratings[k], key=lambda a: float(a["time"]))

    return all_users, all_movies, user_ratings


class MovieTrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.all_users, self.all_movies, self.all_interactions = build_movie_dataset(
            args
        )
        self.train_interactions = {
            k: v[:-2] for k, v in self.all_interactions.items() if len(v) >= 10
        }
        self.all_users = {
            k: v for k, v in self.all_users.items() if k in self.train_interactions
        }
        self.user_indices = list(self.all_users.keys())
        self.movie_indices = list(self.all_movies.keys())
        self.number_of_movie_types()
        self.number_of_interactions()
        self.length = self.__len__()

    # helper function
    def number_of_movie_types(self):
        self.movie_category_indices = []
        types = []
        for k, v in self.all_movies.items():
            types += v["category"]
            for category in v["category"]:
                self.movie_category_indices.append([k, category, "pos"])
                for _ in range(self.args.movie_category_negative_sample):
                    self.movie_category_indices.append([k, category, "neg"])
        self.movie_types = set(types)
        self.movie_types_num = len(set(types))

    # helper function
    def number_of_movie_question(self):
        total_num = 0
        for _, v in self.all_movies.items():
            total_num += len(v["category"])
        return total_num

    # helper function
    def number_of_interactions(self):
        self.user_interaction_code = []
        total_num = 0
        for k, v in self.train_interactions.items():
            number = (
                len(v) - (len(v) % self.args.direct_item_proportion)
            ) / self.args.direct_item_proportion
            total_num += int(number)
            for _ in range(int(number)):
                self.user_interaction_code.append(k)
        return total_num

    def number_of_direct_rec(self):
        total_num = self.number_of_interactions()
        total_num = int(total_num / 2)

        return total_num

    def __len__(self):
        # (1) generate movie types
        # (2) yes/no question about movie type: one positive, one negative
        self.movie_number = (
            len(self.all_movies)
            * len(
                [
                    template
                    for template in task_movie_information
                    if template["task"] == "movie-type"
                ]
            )
            + (1 + self.args.movie_category_negative_sample)
            * self.number_of_movie_question()
        )
        # for each user, generate sequential_num sequences, with some different templates
        self.sequential_item_number = (
            len(self.all_users)
            * self.args.sequential_num
            * len(
                [
                    task
                    for task in task_sequential_movie
                    if task["target_argv"][0] == "item"
                ]
            )
        )
        # for each user, generate its whole sequences, with some different templates
        self.sequential_sequence_number = len(self.all_users) * len(
            [
                task
                for task in task_sequential_movie
                if task["target_argv"][0] == "sequence"
            ]
        )
        # for each user, given a sequence, generate the rest of movies, with some different templates
        self.sequential_subsequence_number = (
            len(self.all_users)
            * self.args.sequential_num
            * len(
                [
                    task
                    for task in task_sequential_movie
                    if task["target_argv"][0] == "subsequence"
                ]
            )
        )
        # for each user, given a sequence, generate the previous movies, with some different templates
        self.sequential_presequence_number = (
            len(self.all_users)
            * self.args.sequential_num
            * len(
                [
                    task
                    for task in task_sequential_movie
                    if task["target_argv"][0] == "presequence"
                ]
            )
        )
        # for each user, given a sequence one potential movie, decide whether this is watched, with some different templates
        self.sequential_yn_number = (
            len(self.all_users)
            * self.args.sequential_num
            * len(
                [
                    task
                    for task in task_sequential_movie
                    if task["target_argv"][0] == "sequential_yes_no"
                ]
            )
        ) * (1 + self.args.negative_sample)
        # for each interaction, choose the movie watched among candidates
        self.direct_item_number = self.number_of_direct_rec() * len(
            [
                task
                for task in task_direct_movie
                if task["target_argv"][0] == "groundtruth_item_ids"
            ]
        )
        # for each user, for a candidate, decide whether it's watched
        self.direct_yn_number = (
            (
                len(self.all_users)
                * len(
                    [
                        task
                        for task in task_direct_movie
                        if task["target_argv"][0] == "yes_no"
                    ]
                )
            )
            * (1 + self.args.negative_sample)
            * self.args.yes_no_sample
        )
        length = (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
            + self.direct_item_number
            + self.direct_yn_number
        )

        return length

    def get_movie_item(self, start, index):
        index -= start
        movie_type_number = len(
            [t for t in task_movie_information if t["task"] == "movie-type"]
        )
        # 7766
        if index < movie_type_number * len(self.all_movies):
            movie_idx = index // movie_type_number
            template_idx = index % movie_type_number
            template = [t for t in task_movie_information if t["task"] == "movie-type"][
                template_idx
            ]
            input_sent = template["source"].format(self.movie_indices[movie_idx])
            output_sent = template["target"].format(
                " , ".join(self.all_movies[self.movie_indices[movie_idx]]["category"])
            )
            return input_sent, output_sent
        else:
            template = [
                t for t in task_movie_information if t["task"] == "movie-yes/no"
            ][0]
            index -= movie_type_number * len(self.all_movies)
            # movie_idx, category_idx, pos/neg
            movie_idx, category, polarity = self.movie_category_indices[index]
            categories = self.all_movies[movie_idx]["category"]
            if polarity == "pos":
                input_sent = template["source"].format(movie_idx, category)
                output_sent = template["target"].format(movie_idx, "", category)
                return input_sent, output_sent
            else:
                category = random.choice(list(self.movie_types - set(categories)))
                input_sent = template["source"].format(movie_idx, category)
                output_sent = template["target"].format(movie_idx, "not", category)
                return input_sent, output_sent

    def get_sequential_item(self, start, index):
        index -= start
        templates = [
            task for task in task_sequential_movie if task["target_argv"][0] == "item"
        ]
        non_random_index = int(index / self.args.sequential_num)
        user_idx = self.user_indices[non_random_index // len(templates)]
        template_idx = non_random_index % len(templates)
        sequence = [
            interaction["movie"]["index"]
            for interaction in self.train_interactions[user_idx]
        ]
        end_candidates = [
            _
            for _ in range(
                max(2, len(sequence) - self.args.sequential_num - 1), len(sequence) - 1,
            )
        ]
        end_index = random.randint(0, len(end_candidates) - 1)
        end_pos = end_candidates[end_index]
        start_candidates = [_ for _ in range(1, min(4, end_pos))]
        start_index = random.randint(0, len(start_candidates) - 1)
        start_pos = start_candidates[start_index]
        purchase_history = sequence[start_pos : end_pos + 1]
        if len(purchase_history) > self.args.max_history:
            purchase_history = purchase_history[-self.args.max_history :]
        target_item = sequence[end_pos + 1]

        template = templates[template_idx]

        input_sent = template["source"].format(user_idx, " , ".join(purchase_history))
        output_sent = template["target"].format(target_item)

        return input_sent, output_sent

    def get_sequential_sequence(self, start, index):
        index -= start
        templates = [
            task
            for task in task_sequential_movie
            if task["target_argv"][0] == "sequence"
        ]
        user_idx = self.user_indices[index // len(templates)]
        template_idx = index % len(templates)

        sequence = [
            interaction["movie"]["index"]
            for interaction in self.train_interactions[user_idx]
        ]

        template = templates[template_idx]
        input_sent = template["source"].format(user_idx)
        output_sent = template["target"].format(" , ".join(sequence))

        return input_sent, output_sent

    def get_sequential_subsequence(self, start, index):
        index -= start
        templates = [
            task
            for task in task_sequential_movie
            if task["target_argv"][0] == "subsequence"
        ]

        non_random_index = int(index / self.args.sequential_num)
        user_idx = self.user_indices[non_random_index // len(templates)]
        template_idx = non_random_index % len(templates)

        sequence = [
            interaction["movie"]["index"]
            for interaction in self.train_interactions[user_idx]
        ]
        end_candidates = [
            _
            for _ in range(
                max(2, len(sequence) - self.args.sequential_num - 1), len(sequence) - 1,
            )
        ]
        end_index = random.randint(0, len(end_candidates) - 1)
        end_pos = end_candidates[end_index]
        presequence = sequence[:end_pos][-self.args.max_history :]
        target_subsequence = sequence[end_pos:]

        template = templates[template_idx]
        input_sent = template["source"].format(user_idx, " , ".join(presequence))
        output_sent = template["target"].format(" , ".join(target_subsequence))

        return input_sent, output_sent

    def get_sequential_presequence(self, start, index):
        index -= start
        templates = [
            task
            for task in task_sequential_movie
            if task["target_argv"][0] == "presequence"
        ]

        non_random_index = int(index / self.args.sequential_num)
        user_idx = self.user_indices[non_random_index // len(templates)]
        template_idx = non_random_index % len(templates)

        sequence = [
            interaction["movie"]["index"]
            for interaction in self.train_interactions[user_idx]
        ]
        begin_candidates = [
            _ for _ in range(1, min(len(sequence) - 1, self.args.sequential_num + 1),)
        ]
        begin_index = random.randint(0, len(begin_candidates) - 1)
        begin_pos = begin_candidates[begin_index]
        subsequence = sequence[begin_pos:][: self.args.max_history]
        target_presequence = sequence[:begin_pos]

        template = templates[template_idx]
        input_sent = template["source"].format(user_idx, " , ".join(subsequence))
        output_sent = template["target"].format(" , ".join(target_presequence))

        return input_sent, output_sent

    def get_sequential_yesno(self, start, index):
        index -= start
        templates = [
            task
            for task in task_sequential_movie
            if task["target_argv"][0] == "sequential_yes_no"
        ]
        polarity = "pos" if index % (1 + self.args.negative_sample) == 0 else "neg"
        index = index // (1 + self.args.negative_sample)
        non_random_index = int(index / self.args.sequential_num)
        user_idx = self.user_indices[non_random_index // len(templates)]
        template_idx = non_random_index % len(templates)
        sequence = [
            interaction["movie"]["index"]
            for interaction in self.train_interactions[user_idx]
        ]
        end_candidates = [
            _
            for _ in range(
                max(2, len(sequence) - self.args.sequential_num * 2 - 1),
                len(sequence) - 1,
            )
        ]
        end_index = random.randint(0, len(end_candidates) - 1)
        end_pos = end_candidates[end_index]
        start_candidates = [_ for _ in range(1, min(4, end_pos))]
        start_index = random.randint(0, len(start_candidates) - 1)
        start_pos = start_candidates[start_index]
        purchase_history = sequence[start_pos : end_pos + 1]
        if len(purchase_history) > self.args.max_history:
            purchase_history = purchase_history[-self.args.max_history :]
        target_item = sequence[end_pos + 1]

        template = templates[template_idx]

        if polarity == "pos":
            input_sent = template["source"].format(
                user_idx, " , ".join(purchase_history), target_item
            )
            output_sent = template["target"].format("yes")
        else:
            candidates = self.movie_indices.copy()
            candidates.remove(target_item)
            negative_item = random.choice(candidates)
            input_sent = template["source"].format(
                user_idx, " , ".join(purchase_history), negative_item
            )
            output_sent = template["target"].format("no")

        return input_sent, output_sent

    def get_direct_item(self, start, index):
        index -= start
        templates = [
            task
            for task in task_direct_movie
            if task["target_argv"][0] == "groundtruth_item_ids"
        ]
        template_idx = index % len(templates)

        index = index // len(templates)
        user_idx = self.user_interaction_code[index]

        positive_items = [
            interaction["movie"]["index"]
            for interaction in self.all_interactions[user_idx]
        ]
        target_item = random.choice(positive_items[:-2])
        negative_items = self.movie_indices.copy()
        negative_items = list(set(negative_items) - set(positive_items))
        negative_items = random.sample(negative_items, k=100)
        candidates = [target_item] + negative_items
        random.shuffle(candidates)

        template = templates[template_idx]
        input_sent = template["source"].format(user_idx, " , ".join(candidates))
        output_sent = template["target"].format(target_item)

        return input_sent, output_sent

    def get_direct_yesno(self, start, index):
        index -= start
        templates = [
            task for task in task_direct_movie if task["target_argv"][0] == "yes_no"
        ]

        index = int(index / self.args.yes_no_sample)

        polarity = "pos" if index % (1 + self.args.negative_sample) == 0 else "neg"
        index = index // (1 + self.args.negative_sample)
        template_idx = index % len(templates)
        user_idx = self.user_indices[index // len(templates)]

        template = templates[template_idx]
        positive_items = [
            interaction["movie"]["index"]
            for interaction in self.all_interactions[user_idx]
        ]
        if polarity == "pos":
            target_item = random.choice(positive_items[:-2])
            input_sent = template["source"].format(user_idx, target_item)
            output_sent = template["target"].format("yes")
        else:
            negative_items = self.movie_indices.copy()
            negative_items = list(set(negative_items) - set(positive_items))
            negative_item = random.choice(negative_items)
            input_sent = template["source"].format(user_idx, negative_item)
            output_sent = template["target"].format("no")

        return input_sent, output_sent

    def __getitem__(self, index):
        if index < self.movie_number:
            start = 0
            input_sent, output_sent = self.get_movie_item(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number <= index
            and index < self.movie_number + self.sequential_item_number
        ):
            start = self.movie_number
            input_sent, output_sent = self.get_sequential_item(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number + self.sequential_item_number <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
        ):
            start = self.movie_number + self.sequential_item_number
            input_sent, output_sent = self.get_sequential_sequence(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
        ):
            start = (
                self.movie_number
                + self.sequential_item_number
                + self.sequential_sequence_number
            )
            input_sent, output_sent = self.get_sequential_subsequence(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
        ):
            start = (
                self.movie_number
                + self.sequential_item_number
                + self.sequential_sequence_number
                + self.sequential_subsequence_number
            )
            input_sent, output_sent = self.get_sequential_presequence(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
        ):
            start = (
                self.movie_number
                + self.sequential_item_number
                + self.sequential_sequence_number
                + self.sequential_subsequence_number
                + self.sequential_presequence_number
            )
            input_sent, output_sent = self.get_sequential_yesno(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
            <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
            + self.direct_item_number
        ):
            start = (
                self.movie_number
                + self.sequential_item_number
                + self.sequential_sequence_number
                + self.sequential_subsequence_number
                + self.sequential_presequence_number
                + self.sequential_yn_number
            )
            input_sent, output_sent = self.get_direct_item(start, index)
            return input_sent, output_sent
        elif (
            self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
            + self.direct_item_number
            <= index
            and index
            < self.movie_number
            + self.sequential_item_number
            + self.sequential_sequence_number
            + self.sequential_subsequence_number
            + self.sequential_presequence_number
            + self.sequential_yn_number
            + self.direct_item_number
            + self.direct_yn_number
        ):
            start = (
                self.movie_number
                + self.sequential_item_number
                + self.sequential_sequence_number
                + self.sequential_subsequence_number
                + self.sequential_presequence_number
                + self.sequential_yn_number
                + self.direct_item_number
            )
            input_sent, output_sent = self.get_direct_yesno(start, index)
            return input_sent, output_sent


class MovieEvalDataset(Dataset):
    def __init__(self, args, mode, template_id):
        super().__init__()
        self.args = args
        self.mode = mode
        self.template_id = template_id
        self.all_users, self.all_movies, self.all_interactions = build_movie_dataset(
            args
        )
        if self.mode == "val":
            self.interactions = {
                k: v[:-1] for k, v in self.all_interactions.items() if len(v) >= 10
            }
        else:
            assert self.mode == "test"
            self.interactions = {
                k: v[:-2] + v[-1:]
                for k, v in self.all_interactions.items()
                if len(v) >= 10
            }
        self.all_users = {
            k: v for k, v in self.all_users.items() if k in self.interactions
        }
        self.user_indices = list(self.all_users.keys())
        self.movie_indices = list(self.all_movies.keys())
        self.length = self.__len__()

    def __len__(self):
        return len(self.all_users)

    def get_sequential_item(self, index):
        template = [
            task for task in task_sequential_movie if task["id"] == self.template_id
        ][0]
        user_idx = self.user_indices[index]

        sequence = [
            interaction["movie"]["index"] for interaction in self.interactions[user_idx]
        ]
        target_item = sequence[-1]
        if len(sequence[:-1]) > self.args.max_history:
            purchase_history = sequence[:-1][-self.args.max_history :]
        else:
            purchase_history = sequence[:-1]

        input_sent = template["source"].format(user_idx, " , ".join(purchase_history))
        output_sent = template["target"].format(target_item)

        return input_sent, output_sent

    def get_direct_item(self, index):
        template = [
            task for task in task_direct_movie if task["id"] == self.template_id
        ][0]

        user_idx = self.user_indices[index]

        positive_items = [
            interaction["movie"]["index"]
            for interaction in self.all_interactions[user_idx]
        ]
        if self.mode == "val":
            target_item = positive_items[-2]
        else:
            assert self.mode == "test"
            target_item = positive_items[-1]
        negative_items = self.movie_indices.copy()
        negative_items = list(set(negative_items) - set(positive_items))
        negative_items = random.sample(negative_items, k=100)
        candidates = [target_item] + negative_items
        random.shuffle(candidates)

        input_sent = template["source"].format(user_idx, " , ".join(candidates))
        output_sent = template["target"].format(target_item)

        return input_sent, output_sent

    def __getitem__(self, index):
        if self.template_id.split("-")[0] == "2":
            input_sent, output_sent = self.get_sequential_item(index)
            return input_sent, output_sent
        else:
            input_sent, output_sent = self.get_direct_item(index)
            return input_sent, output_sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--movie_category_negative_sample", type=int, default=10)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--sequential_num", type=int, default=25)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    parser.add_argument("--index", type=int, default=0)

    args = parser.parse_args()

    dataset = MovieTrainDataset(args)

    # print(len(dataset))

    from tqdm import tqdm
    import time

    print(dataset[args.index])

    # for i in tqdm(range(len(dataset))):
    #    a, b = dataset[i]

    """
    for _ in range(1000):
        i = random.choice(range(len(dataset)))
        a, b = dataset[i]
        print((a, b))
        print("\n")
        time.sleep(10)
    """
