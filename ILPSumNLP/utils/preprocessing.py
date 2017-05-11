#!/usr/bin/python
# -*- coding: utf8 -*-

import stat

from utils.normalizer import *
from utils.utils import *

# preprocess_duc04('../Datasets/DUC04', '../Temp/datasets/en')
# preprocess_vimds('../Datasets/VietnameseMDS', '../Temp/datasets/vi')

ROUGE_PATH = full_path('../Tools/rouge-1.5.5')
ROUGE_2_CSV_PATH = full_path('../Tools/rouge2csv')
BLEU_PATH = full_path('../Tools/bleu-kit-1.05/scripts')


def chmod_exec(file_name):
    file_name = full_path(file_name)
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def make_rouge_script(config, peer_root, model_root, save_path):
    lines = ['<ROUGE_EVAL version="1.5.5">']

    for k, v in config.items():
        lines.append('\t<EVAL ID="%s">' % k)

        lines.append('\t\t<INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>')
        lines.append('\t\t<MODEL-ROOT>%s/%s</MODEL-ROOT>' % (model_root, k))
        lines.append('\t\t<PEER-ROOT>%s/%s</PEER-ROOT>' % (peer_root, k))

        # Models
        lines.append('\t\t<MODELS>')
        for model in v['models']:
            lines.append('\t\t\t<M ID="%s">%s</M>' % (model, model))
        lines.append('\t\t</MODELS>')

        # Peers
        lines.append('\t\t<PEERS>')
        for peer in v['peers']:
            lines.append('\t\t\t<P ID="%s">%s</P>' % (peer, peer))
        lines.append('\t\t</PEERS>')

        lines.append('\t</EVAL>')

    lines.append('</ROUGE_EVAL>')

    # Write xml file
    write_lines(lines, '%s/rouge_config.xml' % save_path)

    # Write sh file
    # 'perl $rouge_file -e $data_path -a -d -c 95 -r 1000 -b 665 -n 2 -m -2 4 -u -f A -p 0.5 -t 0 '
    lines = [
        'rouge_file=%s/ROUGE-1.5.5.pl' % ROUGE_PATH,
        'data_path=%s/data' % ROUGE_PATH,
        'perl $rouge_file -e $data_path -a -d -c 95 -r 1000 -n 2 -m -2 4 -u -f A -p 0.5 -t 0 '
        'rouge_config.xml >> ${1:-rouge_result.out}'
    ]

    run_rouge_file = '%s/run_rouge.sh' % save_path
    write_lines(lines, run_rouge_file)
    chmod_exec(run_rouge_file)


def make_bleu_script(config, peer_root, model_root, save_path):
    lines = ['cmd="ruby %s/doc_bleu.rb --ngram 4"' % BLEU_PATH]

    for k, v in config.items():
        for peer in v['peers']:
            for model in v['models']:
                lines.append('$cmd "%s/%s/%s" "%s/%s/%s"' % (peer_root, k, peer, model_root, k, model))
                lines.append('echo')

    # Write sh file
    bleu_config_file = '%s/bleu_config.sh' % save_path
    write_lines(lines, bleu_config_file)
    chmod_exec(bleu_config_file)

    run_bleu_file = '%s/run_bleu.sh' % save_path
    write_file('./bleu_config.sh >> ${1:-bleu_result.out}\n', run_bleu_file)
    chmod_exec(run_bleu_file)


def preprocess_duc04(dir_path, save_path):
    num_docs = 500
    num_refs = 200

    docs_count = 0
    refs_count = 0

    content_pattern = regex.compile(r'<TEXT>(.+?)<\/TEXT>', regex.DOTALL | regex.IGNORECASE)

    clusters_dir_path = '%s/clusters' % save_path
    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(peers_dir_path)

    mapping = {}
    config = {}
    packed = {}

    ref_docs, refs_path = read_dir('%s/models' % dir_path, dir_filter=True)

    clusters, clusters_path = read_dir('%s/docs' % dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        cluster_name = 'cluster_%d' % (i + 1)

        # Config
        config[cluster_name] = {
            'models': [],
            'peers': ['1']
        }

        # Mapping
        mapping[cluster_name] = {
            'original_name': cluster,
            'models': {}
        }

        # Pack
        packed[cluster_name] = {
            'docs': [],
            'path': '%s/%s/1' % (peers_dir_path, cluster_name)
        }

        # Docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            file_name = '%s/%s/%d' % (clusters_dir_path, cluster_name, j + 1)
            file_content = read_file(docs_path[j])
            matcher = content_pattern.search(file_content)
            if matcher is not None:
                file_content = matcher.group(1)

                # Preprocessing
                file_content = normalize_dataset(file_content)
                packed[cluster_name]['docs'].append(file_content)

                write_file(file_content, file_name)

                docs_count += 1

        # Refs
        ref_id = 0
        for j, ref_doc in enumerate(ref_docs):
            if cluster.lower()[:-1] in ref_doc.lower():
                config[cluster_name]['models'].append(str(ref_id + 1))
                mapping[cluster_name]['models'][ref_id + 1] = ref_doc

                file_name = '%s/%s/%d' % (models_dir_path, cluster_name, ref_id + 1)
                file_content = read_file(refs_path[j])

                # Preprocessing
                write_file(normalize_dataset(file_content), file_name)

                ref_id += 1
                refs_count += 1

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

    write_json(packed, '%s/packed.json' % save_path)
    write_json(mapping, '%s/mapping.json' % save_path)
    make_rouge_script(config, 'peers', 'models', save_path)
    make_bleu_script(config, 'peers', 'models', save_path)


def preprocess_vimds(dir_path, save_path):
    num_docs = 628
    num_refs = 398

    docs_count = 0
    refs_count = 0

    clusters_dir_path = '%s/clusters' % save_path
    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(peers_dir_path)

    mapping = {}
    config = {}
    packed = {}

    clusters, clusters_path = read_dir(dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        cluster_name = 'cluster_%d' % (i + 1)

        # Config
        config[cluster_name] = {
            'models': [],
            'peers': ['1']
        }

        # Mapping
        mapping[cluster_name] = {
            'original_name': cluster,
            'models': {}
        }

        # Pack
        packed[cluster_name] = {
            'docs': [],
            'path': '%s/%s/1' % (peers_dir_path, cluster_name)
        }

        # Docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)

        doc_id = 0
        ref_id = 0

        for j, doc in enumerate(docs):
            doc_name = doc.lower()
            if '.body.txt' in doc_name:  # Doc
                file_name = '%s/cluster_%d/%d' % (clusters_dir_path, i + 1, doc_id + 1)
                file_content = read_file(docs_path[j])

                # Preprocessing
                file_content = normalize_dataset(file_content, lang='vi')
                packed[cluster_name]['docs'].append(file_content)

                write_file(file_content, file_name)

                doc_id += 1
                docs_count += 1

            elif '.ref' in doc_name and '.tok' not in doc_name:  # Ref
                config[cluster_name]['models'].append(str(ref_id + 1))
                mapping[cluster_name]['models'][ref_id + 1] = doc_name

                file_name = '%s/cluster_%d/%d' % (models_dir_path, i + 1, ref_id + 1)
                file_content = read_file(docs_path[j])

                # Preprocessing
                write_file(normalize_dataset(file_content, lang='vi'), file_name)

                ref_id += 1
                refs_count += 1

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

    write_json(packed, '%s/packed.json' % save_path)
    write_json(mapping, '%s/mapping.json' % save_path)
    make_rouge_script(config, 'peers', 'models', save_path)
    make_bleu_script(config, 'peers', 'models', save_path)
