#!/usr/bin/python
# -*- coding: utf8 -*-

import shutil
import stat

from utils.normalizer import *
from utils.utils import *

ROUGE_PATH = full_path('../Tools/rouge-1.5.5')


# OK
def chmod_exec(file_name):
    file_name = full_path(file_name)
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


# OK
def make_rouge_script(data_info, peer_root, model_root, save_path):
    # Header file
    lines = ['<ROUGE_EVAL version="1.5.5">']

    # Body file
    for info in data_info:
        # Begin of segment
        lines.append('\t<EVAL ID="%s">' % info['name'])

        # Configs
        lines.append('\t\t<INPUT-FORMAT TYPE="SPL"></INPUT-FORMAT>')
        lines.append('\t\t<MODEL-ROOT>%s</MODEL-ROOT>' % model_root)
        lines.append('\t\t<PEER-ROOT>%s</PEER-ROOT>' % peer_root)

        # Models
        lines.append('\t\t<MODELS>')
        for model in info['models']:
            lines.append('\t\t\t<M ID="%s">%s</M>' % (model['name'], model['file']))
        lines.append('\t\t</MODELS>')

        # Peers
        lines.append('\t\t<PEERS>')
        for peer in info['peers']:
            lines.append('\t\t\t<P ID="%s">%s</P>' % (peer['name'], peer['file']))
        lines.append('\t\t</PEERS>')

        # End of segment
        lines.append('\t</EVAL>')

    # Footer file
    lines.append('</ROUGE_EVAL>')

    # Write xml file
    write_lines(lines, '%s/rouge_config.xml' % save_path)

    # Write sh file
    # 'perl $rouge_file -e $data_path -a -d -c 95 -r 1000 -l 100 -n 2 -m -2 4 -u -f A -p 0.5 -t 0 '
    lines = [
        'rouge_path=%s' % ROUGE_PATH,
        'rouge_file=$rouge_path/ROUGE-1.5.5.pl',
        'data_path=$rouge_path/data',
        'output_file=${1:-rouge_result.out}',
        'rm $output_file',
        'perl $rouge_file -e $data_path -a -d -c 95 -r 1000 -n 2 -m -2 4 -u -f A -p 0.5 -t 0 '
        'rouge_config.xml >> $output_file'
    ]

    run_rouge_file = '%s/run_rouge.sh' % save_path
    write_lines(lines, run_rouge_file)
    chmod_exec(run_rouge_file)


# OK
def preprocess_duc04(dir_path, save_path):
    # Actual size for checking
    num_docs = 500
    num_refs = 200

    # For counting
    docs_count = 0
    refs_count = 0

    content_pattern = regex.compile(r'<TEXT>(.+?)<\/TEXT>', regex.DOTALL | regex.IGNORECASE)

    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(models_dir_path)
    make_dirs(peers_dir_path)

    # For storing clusters
    data_info = []

    ref_docs, refs_path = read_dir('%s/models' % dir_path, dir_filter=True)
    clusters, clusters_path = read_dir('%s/docs' % dir_path, file_filter=True)

    # Iterate over the clusters
    for i, cluster in enumerate(clusters):
        cluster_info = {
            'name': cluster,
            'docs': [],
            'models': [],
            'peers': [{
                'name': 'ILPSum',
                'file': cluster
            }],
            'save': '%s/%s' % (peers_dir_path, cluster)
        }

        # Iterate over the docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            matcher = content_pattern.search(read_file(docs_path[j]))
            if matcher is not None:
                # Preprocessing
                cluster_info['docs'].append({
                    'name': doc,
                    'content': matcher.group(1)
                })
                docs_count += 1

        # Iterate over the models
        ref_id = 0
        for j, ref_doc in enumerate(ref_docs):
            if cluster.lower()[:-1] in ref_doc.lower():
                # Preprocessing
                cluster_info['models'].append({
                    'name': ref_doc[-1],
                    'file': ref_doc,
                    'num_words': num_words(read_file(refs_path[j]))
                })

                # Copy model file
                shutil.copy(src=full_path(refs_path[j]), dst=full_path(models_dir_path))

                ref_id += 1
                refs_count += 1

        data_info.append(cluster_info)

    assert docs_count == num_docs, 'There should be the same number of documents in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

    # Write config file
    write_json(data_info, '%s/input.json' % save_path)
    make_rouge_script(data_info, 'peers', 'models', save_path)


def preprocess_vimds(dir_path, save_path):
    # Actual size for checking
    num_docs = 628
    num_refs = 398

    # For counting
    docs_count = 0
    refs_count = 0

    clusters_dir_path = '%s/clusters' % save_path
    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(peers_dir_path)

    data_info = []

    clusters, clusters_path = read_dir(dir_path, file_filter=True)

    for i, cluster in enumerate(clusters):
        cluster_name = 'cluster_%d' % (i + 1)

        info = {
            'cluster_name': cluster_name,
            'original_cluster_name': cluster,
            'docs': [],
            'models': [],
            'peers': [('ILPSum', str(i + 1))],
            'save': '%s/%s' % (peers_dir_path, i + 1)
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
                info['docs'].append({
                    'doc_name': str(doc_id + 1),
                    'original_name': doc,
                    'content': remove_invalid_chars(file_content)
                })

                write_file(' '.join(normalize_dataset(file_content, lang='vi')), file_name)

                doc_id += 1
                docs_count += 1

            elif '.ref' in doc_name and '.tok' not in doc_name:  # Ref
                file_name = '%s/cluster_%d/%d' % (models_dir_path, i + 1, ref_id + 1)
                file_content = read_file(docs_path[j])

                # Preprocessing
                tokens = normalize_dataset(file_content, lang='vi')
                write_file(' '.join(tokens), file_name)

                info['models'].append({
                    'model_name': str(ref_id + 1),
                    'original_name': doc_name,
                    'num_words': len(remove_punctuation(tokens))
                })

                ref_id += 1
                refs_count += 1

        data_info.append(info)

    assert docs_count == num_docs, 'There should be the same number of docs in clusters'
    assert refs_count == num_refs, 'There should be the same number of reference documents in clusters'

    write_json(data_info, '%s/info.json' % save_path)
    make_rouge_script(data_info, 'peers', 'models', save_path)
