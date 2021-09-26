from codecs import iterdecode
from platform import node
from numpy.core import einsumfunc
import os
import warnings
import re
import random
from typing import List
import json
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
import csv

from .util import prepare_dirs_and_logger
from .config import AnalysisArguments, DataArguments, MiscArgument, get_config, SourceMap, TrustMap, ArticleMap, FullArticleMap

warnings.filterwarnings('ignore')


def _clean_text(original_tweet: str) -> str:
    processed_tweet = re.sub(r'http[^ ]+', 'URL', original_tweet)
    processed_tweet = re.sub(r'RT @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'rt @[^ ]+ ', '', processed_tweet)
    processed_tweet = re.sub(r'@\S+$|@\S+ ', '', processed_tweet)
    processed_tweet = processed_tweet.replace('\n', ' ')
    processed_tweet = processed_tweet.replace('\r', '')
    processed_tweet = processed_tweet.replace('RT', '')
    processed_tweet = processed_tweet.replace('rt', '')
    processed_tweet = re.sub(r' +', ' ', processed_tweet)
    processed_tweet = processed_tweet.strip()
    return processed_tweet


def article_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    article_map = ArticleMap()

    article_dict = dict()
    data_path_dir_list_temp = []
    data_path_dir_list = []
    file_path_list = []

    train_topic_list = ['abortion', 'marijuana', 'drug policy', 'gay marriage']
    eval_topic_list = ['corporate tax']
    test_topic_list = ['climate change', 'obamacare']

    topic_dict = {'train': train_topic_list,
                  'eval': eval_topic_list, 'test': test_topic_list}

    if data_args.dataset in topic_dict:
        topic_list = topic_dict[data_args.dataset]
    else:
        topic_list = [data_args.dataset]
    data_args.data_dir = data_args.data_dir + '_' + data_args.dataset

    year_list = os.listdir(data_args.original_data_dir)
    for year in year_list:
        data_path_dir = os.path.join(data_args.original_data_dir, year)
        data_path_dir_list_temp.append(data_path_dir)

    for data_path_year in data_path_dir_list_temp:
        # topic_list = os.listdir(data_path_year)
        for topic in topic_list:
            data_path_dir = os.path.join(data_path_year, topic)
            data_path_dir_list.append(data_path_dir)

    result_list = []
    if misc_args.global_debug:
        for data_path_dir in data_path_dir_list:
            article_dict_temp = _article_collect(
                data_path_dir, misc_args.global_debug)
            result_list.append(article_dict_temp)
    else:
        with Pool(processes=10) as pool:
            for data_path_dir in data_path_dir_list:
                article_dict_temp = pool.apply_async(
                    func=_article_collect, args=(data_path_dir, misc_args.global_debug,))
                result_list.append(article_dict_temp)
            pool.close()
            pool.join()

    for result in result_list:
        if misc_args.global_debug:
            result = result
        else:
            result = result.get()
        for media, text_list in result.items():
            if media not in article_dict:
                article_dict[media] = list()
            article_dict[media].extend(text_list)

    for media_name, text_list in article_dict.items():
        if media_name not in article_map.name_to_dataset:
            continue
        media = article_map.name_to_dataset[media_name]
        text_list = list(set(text_list))
        random.shuffle(text_list)
        train_number = int(len(text_list)*0.7)
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in text_list[:train_number]:
                if len(text) < 5:
                    continue
                if len(text) > 512:
                    start_index = 0
                    end_index = 0
                    while start_index < len(text):
                        text_t = text[start_index:start_index+512]
                        end_index = text_t.rfind('.')
                        if end_index != -1:
                            text_t = text_t[:end_index+1]
                        text_t = text_t.strip()
                        fp.write(text_t+'\n')
                        if end_index != -1:
                            start_index += (end_index+1)
                        else:
                            start_index += 512
                else:
                    fp.write(text+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in text_list[train_number:]:
                if len(text) < 5:
                    continue
                if len(text) > 512:
                    start_index = 0
                    end_index = 0
                    while start_index < len(text):
                        text_t = text[start_index:start_index+512]
                        end_index = text_t.rfind('.')
                        if end_index != -1:
                            text_t = text_t[:end_index+1]
                        text_t.strip()
                        fp.write(text_t+'\n')
                        if end_index != -1:
                            start_index += (end_index+1)
                        else:
                            start_index += 512
                else:
                    fp.write(text+'\n')


def _article_collect(data_path_dir, global_debug):
    article_dict = dict()
    file_path_list = os.listdir(data_path_dir)
    if global_debug:
        file_path_list = file_path_list[:30]
    for file in file_path_list:
        file_path = os.path.join(data_path_dir, file)
        with open(file_path, mode='r', encoding='utf8') as fp:
            item = json.load(fp)
            text = item['text']
            media = item['media']
        if media not in article_dict:
            article_dict[media] = list()
        article_dict[media].append(_clean_text(text))
    return article_dict


def origianl_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    article_map = FullArticleMap()

    article_dict = dict()
    data_path_dir_list_temp = []
    data_path_dir_list = []

    train_topic_list = ['abortion', 'marijuana', 'drug policy', 'gay marriage']
    eval_topic_list = ['corporate tax']
    test_topic_list = ['climate change', 'obamacare']

    topic_dict = {'train': train_topic_list,
                  'eval': eval_topic_list, 'test': test_topic_list}

    if data_args.dataset in topic_dict:
        topic_list = topic_dict[data_args.dataset]
    else:
        topic_list = [data_args.dataset.replace('-', ' ')]
    data_args.data_dir = data_args.data_dir + '_' + data_args.dataset

    year_list = os.listdir(data_args.original_data_dir)
    for year in year_list:
        data_path_dir = os.path.join(data_args.original_data_dir, year)
        data_path_dir_list_temp.append(data_path_dir)

    for data_path_year in data_path_dir_list_temp:
        for topic in topic_list:
            data_path_dir = os.path.join(data_path_year, topic)
            data_path_dir_list.append(data_path_dir)

    result_list = []
    if misc_args.global_debug:
        for data_path_dir in data_path_dir_list:
            article_dict_temp = _original_collect(
                data_path_dir, misc_args.global_debug)
            result_list.append(article_dict_temp)
    else:
        with Pool(processes=10) as pool:
            for data_path_dir in data_path_dir_list:
                article_dict_temp = pool.apply_async(
                    func=_original_collect, args=(data_path_dir, misc_args.global_debug,))
                result_list.append(article_dict_temp)
            pool.close()
            pool.join()

    for result in result_list:
        if misc_args.global_debug:
            result = result
        else:
            result = result.get()
        for media, text_list in result.items():
            if media not in article_dict:
                article_dict[media] = list()
            article_dict[media].extend(text_list)

    for media_name, text_list in article_dict.items():
        if media_name not in article_map.name_to_dataset:
            continue
        media = article_map.name_to_dataset[media_name]
        text_list = list(set(text_list))
        if len(text_list) < 50:
            continue
        random.shuffle(text_list)
        train_number = int(len(text_list)*0.9)
        data_path = os.path.join(os.path.join(
            data_args.data_dir, media), data_args.data_type)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        train_file = os.path.join(data_path, 'en.train')
        with open(train_file, mode='w', encoding='utf8') as fp:
            for text in text_list[:train_number]:
                if len(text) < 5:
                    continue
                fp.write(text+'\n')
        eval_file = os.path.join(data_path, 'en.valid')
        with open(eval_file, mode='w', encoding='utf8') as fp:
            for text in text_list[train_number:]:
                if len(text) < 5:
                    continue
                fp.write(text+'\n')


def _original_collect(data_path_dir, global_debug):
    article_dict = dict()
    file_path_list = os.listdir(data_path_dir)
    if global_debug:
        file_path_list = file_path_list[:30]
    for file in file_path_list:
        file_path = os.path.join(data_path_dir, file)
        with open(file_path, mode='r', encoding='utf8') as fp:
            item = json.load(fp)
            text = item['text']
            media = item['media']
        if media not in article_dict:
            article_dict[media] = list()
        text = text.strip().replace('\n', '\\n').replace('\"', '')
        text = text.lower()
        article_dict[media].append(text)
    return article_dict

def all_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    sequence_length = 256
    article_map = ArticleMap()
    raw_data = dict()
    train_data = dict()
    eval_data = dict()

    media_list = article_map.dataset_list
    for media in media_list:
        if media not in raw_data:
            raw_data[media] = dict()
        grouped_train_data = list()
        grouped_eval_data = list()

        train_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.train')
        eval_file = os.path.join(os.path.join(os.path.join(
            data_args.data_dir, media), 'original'), 'en.valid')
        with open(train_file, mode='r', encoding='utf8') as fp:
            for line in fp:
                paragraph_list = line.strip().split('\\n\\n')
                for paragraph in paragraph_list:
                    if len(paragraph.split(' ')) < sequence_length and len(paragraph.split(' ')) > 5:
                        grouped_train_data.append(paragraph)
                    elif len(paragraph.split(' ')) >= sequence_length:
                        sentence_list = sent_tokenize(paragraph.strip())
                        chunk_sentences = str()
                        for sentence in sentence_list:
                            if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < sequence_length:
                                chunk_sentences = chunk_sentences + ' ' + sentence
                            else:
                                grouped_train_data.append(
                                    chunk_sentences.strip())
                                chunk_sentences = sentence
                        grouped_train_data.append(chunk_sentences.strip())
        with open(eval_file, mode='r', encoding='utf8') as fp:
            for line in fp:
                paragraph_list = line.strip().split('\\n\\n')
                for paragraph in paragraph_list:
                    if len(paragraph.split(' ')) < sequence_length and len(paragraph.split(' ')) > 5:
                        grouped_eval_data.append(paragraph)
                    elif len(paragraph.split(' ')) >= sequence_length:
                        sentence_list = sent_tokenize(paragraph.strip())
                        chunk_sentences = str()
                        for sentence in sentence_list:
                            if len(chunk_sentences.split(' ')) + len(sentence.split(' ')) < sequence_length:
                                chunk_sentences = chunk_sentences + ' ' + sentence
                            else:
                                grouped_eval_data.append(
                                    chunk_sentences.strip())
                                chunk_sentences = sentence
                        grouped_eval_data.append(chunk_sentences.strip())
        raw_data[media] = {'train': grouped_train_data,
                           'eval': grouped_eval_data}

    data_path = os.path.join(os.path.join(
        data_args.data_dir,  'all'), 'original')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_file = os.path.join(data_path, 'en.train')
    with open(train_file, mode='w', encoding='utf8') as fp:
        for media in media_list:
            label = article_map.dataset_list.index(media)
            for item in raw_data[media]['train']:
                it = {'sentence':item,'label':label}
                fp.write(json.dumps(it,ensure_ascii=False)+'\n')
            
    eval_file = os.path.join(data_path, 'en.valid')
    with open(eval_file, mode='w', encoding='utf8') as fp:
        for media in media_list:
            label = article_map.dataset_list.index(media)
            for item in raw_data[media]['eval']:
                it = {'sentence':item,'label':label}
                fp.write(json.dumps(it,ensure_ascii=False)+'\n')


def data_collect(
    misc_args: MiscArgument,
    data_args: DataArguments
) -> None:
    if data_args.data_type == 'original':
        origianl_collect(misc_args, data_args)
    elif data_args.data_type == 'all':
        all_collect(misc_args,data_args)


def main():
    misc_args, model_args, data_args, training_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args)
    # twitter_collect(misc_args, data_args)
    # article_collect(misc_args, data_args)
    data_collect(misc_args, data_args)


if __name__ == '__main__':
    main()
