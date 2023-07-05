import pandas as pd
import numpy as np
import pyranges as pr
import anndata
import os


class AnnotationParams:
    columns = [
        'Chromosome', 'Start', 'End', 'Strand', 
        'Feature', 'gene_id', 'gene_type', 'gene_name'
    ]
    features = ['gene', 'exon']
    gene_types = ['protein_coding', 'lncRNA']


def read_gtf(file_name: str) -> pr.PyRanges:
    '''
    Customizations on the columns, features and gene types
    to be read can be manipulated on the AnnotationParams
    object.
    '''
    annotation = pr.read_gtf(file_name)
    
    annotation.Start = annotation.Start.astype(np.int64)
    annotation.End = annotation.End.astype(np.int64)
    
    if AnnotationParams.columns:
        annotation = annotation[AnnotationParams.columns]

    if AnnotationParams.features:
        features = AnnotationParams.features
        annotation = annotation[annotation.Feature.isin(features)]
        
    if AnnotationParams.gene_types:
        gene_types = AnnotationParams.gene_types
        annotation = annotation[annotation.gene_type.isin(gene_types)]
    
    return annotation


def compute_bulk_size_factors(files: list,
                              col_start = 1,
                              col_end = 2,
                              method:str = 'lines',
                              threads: int = 1) -> dict:
    '''
    files: list, tuple, dict, str
        The files need to be in bed format, compressed with bgzip 
        (samtools, .bgzf format) and indexed with tabix (samtools).
    col_start = 1
        Python index: 1 = second column
    col_end = 2
        Python index: 2 = third column
    method: str = "lines"
        "lines" | "bases"
    threads: int = 1
        Possibly will run faster on single theread.
    -----
    Returns: dict
        Relative factors (between files), not absolute (example: 
        group 1 = 0.1 not equal to group 1 constitute 0.1 of the 
        whole read set).
        size factor = read count (group) / mean read count (all groups)
    '''

    import pysam

    def process_files(files) -> dict:
        if not isinstance(files, dict):
            if isinstance(files, str):
                name = os.path.basename(files).split('.')[0]
                files = {name: files}

            if (isinstance(files, tuple) or 
                isinstance(files, list)):
                names = []
                for file in files:
                    names.append(os.path.basename(file).split('.')[0])
                items = zip(names, files)
                files = {name: bgzf_file for name, bgzf_file in items}

            return files

    files = process_files(files)

    def extract_n_lines(file) -> int:
        tbx = pysam.TabixFile(file)
        n_lines = 0
        for i, read in enumerate(tbx.fetch(parser = pysam.asBed())):
            n_lines += i
        return n_lines

    def extract_n_bases(file) -> int:
        tbx = pysam.TabixFile(file)
        n_bases = 0
        for i, read in enumerate(tbx.fetch(parser = pysam.asBed())):
            n_bases += int(read[col_end]) - int(read[col_start])
        return n_bases

    if method not in ["lines", "bases"]:
        f'Method must be "lines" or "bases", not {method}'

    methods = {
        "lines": extract_n_lines,
        "bases": extract_n_bases
    }

    if threads > 1:
        import concurrent.futures
        with (concurrent.futures
             .ThreadPoolExecutor(
             max_workers = threads
             ) as executor):
            counts = list(executor.map(
                methods[method], 
                files.values()
            ))
            items = zip(files.keys(), counts)
            counts = {name: count for name, count in items}
    else:
        counts = {}
        for name, file in files.items():
            print(name)
            counts[name] = methods[method](file)

    # Compute size factors
    total_counts = 0
    for count in counts.values():
        total_counts += count
    mean_counts = total_counts / len(counts)

    size_factors = {}
    for name, count in counts.items():
        size_factors[name] = count / mean_counts

    return size_factors



def compute_single_cell_size_factors(adata: anndata.AnnData, 
                                     file: str, 
                                     metadata_cols: list) -> dict:
    '''
    Relative factors (between groups), not absolute (example: 
    group 1 = 0.1 not equal to group 1 constitute 0.1 of the 
    whole fragments set).
    size factor = fragment count (group) / mean fragment count (all groups)
    '''
    import pysam
    from collections import Counter

    tbx = pysam.TabixFile(file)

    if isinstance(metadata_cols, str):
        metadata_cols = [metadata_cols]
    metadata = adata.obs.loc[:, metadata_cols]

    # import barcodes
    barcodes = []
    for row in tbx.fetch(parser = pysam.asBed()):
        barcodes.append(row.name)

    # count barcodes
    barcodes = pd.Series(Counter(barcodes))
    barcodes = barcodes.rename('fragments_count')

    metadata = metadata.merge(
        barcodes, 
        how = 'left', 
        left_index = True, 
        right_index = True
    )

    aggregator = pd.NamedAgg(
        column = "fragments_count", 
        aggfunc = "sum"
    )

    # compute size factors
    compute_size_factors_ =         lambda x: x.fragments_count / np.mean(x.fragments_count)

    size_factors = (
        metadata
        .groupby(metadata_cols)
        .aggregate(fragments_count = aggregator)
        .assign(size_factors = compute_size_factors_)
        .to_dict()['size_factors']
    )

    return size_factors

