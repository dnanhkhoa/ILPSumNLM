#!/usr/bin/python
# -*- coding: utf8 -*-

import stat

import regex

from utils.normalizer import *
from utils.utils import *

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
    lines = [
        'rouge_file=%s/ROUGE-1.5.5.pl' % ROUGE_PATH,
        'data_path=%s/data' % ROUGE_PATH,
        'perl $rouge_file -e $data_path -a -d -c 95 -r 1000 -b 665 -n 2 -m -2 4 -u -f A -p 0.5 -t 0 '
        'rouge_config.xml >> ${1:-rouge_result.out}'
    ]

    run_rouge_file = '%s/run_rouge.sh' % save_path
    write_lines(lines, run_rouge_file)
    chmod_exec(run_rouge_file)


def make_bleu_script(config, peer_root, model_root, save_path):
    lines = ['cmd="ruby %s/doc_bleu.rb"' % BLEU_PATH]

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

    config = {}

    ref_docs, refs_path = read_dir('%s/models' % dir_path, dir_filter=True)

    clusters, clusters_path = read_dir('%s/docs' % dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        # Config
        config['cluster_%d' % (i + 1)] = {'models': [], 'peers': ['1']}

        # Docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            file_name = '%s/cluster_%d/%d' % (clusters_dir_path, i + 1, j + 1)
            file_content = read_file(docs_path[j])
            matcher = content_pattern.search(file_content)
            if matcher is not None:
                file_content = matcher.group(1)
                # Preprocessing
                write_file(normalize_dataset(file_content), file_name)

                docs_count += 1

        # Refs
        ref_id = 0
        for j, ref_doc in enumerate(ref_docs):
            if cluster.lower()[:-1] in ref_doc.lower():
                config['cluster_%d' % (i + 1)]['models'].append(str(ref_id + 1))

                file_name = '%s/cluster_%d/%d' % (models_dir_path, i + 1, ref_id + 1)
                file_content = read_file(refs_path[j])
                # Preprocessing
                write_file(normalize_dataset(file_content), file_name)
                ref_id += 1
                refs_count += 1

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

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

    config = {}

    clusters, clusters_path = read_dir(dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        # Config
        config['cluster_%d' % (i + 1)] = {'models': [], 'peers': ['1']}

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
                write_file(normalize_dataset(file_content), file_name)
                doc_id += 1
                docs_count += 1

            elif '.ref' in doc_name and '.tok' not in doc_name:  # Ref
                config['cluster_%d' % (i + 1)]['models'].append(str(ref_id + 1))

                file_name = '%s/cluster_%d/%d' % (models_dir_path, i + 1, ref_id + 1)
                file_content = read_file(docs_path[j])
                # Preprocessing
                write_file(normalize_dataset(file_content), file_name)
                ref_id += 1
                refs_count += 1

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

    make_rouge_script(config, 'peers', 'models', save_path)
    make_bleu_script(config, 'peers', 'models', save_path)