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

                num_docs -= 1

        # Iterate over the models
        for j, ref_doc in enumerate(ref_docs):
            if cluster.lower()[:-1] in ref_doc.lower():
                # Preprocessing
                cluster_info['models'].append({
                    'name': ref_doc[-1],
                    'file': ref_doc,
                    'num_words': get_num_words(read_file(refs_path[j]))
                })

                # Copy model file
                shutil.copy(src=full_path(refs_path[j]), dst=full_path(models_dir_path))

                num_refs -= 1

        data_info.append(cluster_info)

    assert num_docs == 0, 'There should be the same number of documents in clusters'
    assert num_refs == 0, 'There should be the same number of reference documents in clusters'

    # Write config file
    write_json(data_info, '%s/input.json' % save_path)
    make_rouge_script(data_info, 'peers', 'models', save_path)


# OK
def preprocess_vimds(dir_path, save_path):
    # Actual size for checking
    num_docs = 628
    num_refs = 398

    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(models_dir_path)
    make_dirs(peers_dir_path)

    # For storing clusters
    data_info = []

    clusters, clusters_path = read_dir(dir_path, file_filter=True)

    # Iterate over the clusters
    for i, cluster in enumerate(clusters):
        raw_docs = []

        # Iterate over the docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            if '.body.txt' in doc.lower():
                # Preprocessing
                raw_docs.append({
                    'name': doc,
                    'content': read_file(docs_path[j])
                })

                num_docs -= 1

        # Iterate over the models
        for j, doc in enumerate(docs):
            doc_name = doc.lower()
            if '.ref' in doc_name and '.tok' not in doc_name:
                # Preprocessing
                ref_id = doc.split('.')[1][-1]

                data_info.append({
                    'name': '%s.%s' % (cluster, ref_id),
                    'docs': raw_docs,
                    'models': [{
                        'name': doc.split('.')[1],
                        'file': doc,
                        'num_words': get_num_words(read_file(docs_path[j]))
                    }],
                    'peers': [{
                        'name': 'ILPSum',
                        'file': '%s.%s' % (cluster, ref_id)
                    }],
                    'save': '%s/%s.%s' % (peers_dir_path, cluster, ref_id)
                })

                # Copy model file
                shutil.copy(src=full_path(docs_path[j]), dst=full_path(models_dir_path))

                num_refs -= 1

    assert num_docs == 0, 'There should be the same number of docs in clusters'
    assert num_refs == 0, 'There should be the same number of reference documents in clusters'

    # Write config file
    write_json(data_info, '%s/input.json' % save_path)
    make_rouge_script(data_info, 'peers', 'models', save_path)


# OK
def preprocess_vimds_hcmus(dir_path, save_path):
    # Actual size for checking
    num_docs = 1955
    num_refs = 629

    header_pattern = regex.compile(r'cluster_\d+\s+=Einleitung=\n+', regex.DOTALL | regex.IGNORECASE)

    models_dir_path = '%s/models' % save_path
    peers_dir_path = '%s/peers' % save_path
    make_dirs(models_dir_path)
    make_dirs(peers_dir_path)

    # For storing clusters
    data_info = []

    clusters, clusters_path = read_dir('%s/docs' % dir_path, file_filter=True)

    # Iterate over the clusters
    for i, cluster in enumerate(clusters):
        raw_docs = []

        # Iterate over the docs
        docs, docs_path = read_dir(clusters_path[i], dir_filter=True)
        for j, doc in enumerate(docs):
            # Preprocessing
            raw_docs.append({
                'name': doc,
                'content': read_file(docs_path[j])
            })

            num_docs -= 1

        # Iterate over the models
        refs, refs_path = read_dir(clusters_path[i].replace('/docs', '/models'), dir_filter=True)
        for j, ref in enumerate(refs):
            # Preprocessing
            ref_id = ref.split('.')[0]

            content = header_pattern.sub('', read_file(refs_path[j]))
            data_info.append({
                'name': '%s.%s' % (cluster, ref_id),
                'docs': raw_docs,
                'models': [{
                    'name': ref_id,
                    'file': '%s.%s' % (cluster, ref),
                    'num_words': get_num_words(content)
                }],
                'peers': [{
                    'name': 'ILPSum',
                    'file': '%s.%s' % (cluster, ref_id)
                }],
                'save': '%s/%s.%s' % (peers_dir_path, cluster, ref_id)
            })

            # Copy model file
            write_file(content, '%s/%s.%s' % (full_path(models_dir_path), cluster, ref))

            num_refs -= 1

    assert num_docs == 0, 'There should be the same number of docs in clusters'
    assert num_refs == 0, 'There should be the same number of reference documents in clusters'

    # Write config file
    write_json(data_info, '%s/input.json' % save_path)
    make_rouge_script(data_info, 'peers', 'models', save_path)
