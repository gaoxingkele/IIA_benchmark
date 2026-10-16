# Single-Cell Multiomics Dataset: Mobilized CD34+ HSPCs Time Course

## Dataset Overview
This dataset contains single-cell multiomics data from mobilized peripheral CD34+ hematopoietic stem and progenitor cells (HSPCs) from four healthy human donors (13176, 27678, 31800, 32606). Cells were cultured over 10 days with measurements at days 2, 3, 4, 7, and 10 using two technologies: 10x Chromium Single Cell Multiome ATAC + Gene Expression (Multiome) and 10x Genomics Single Cell Gene Expression with Feature Barcoding (CITE-seq).

## File Descriptions

### Raw 10x Genomics Output Files

**CITE-seq Raw Data (10x Gene Expression + Feature Barcoding):**
- `SRTP000196[4-9]_filtered_feature_bc_matrix.h5` - 10x filtered count matrices for CITE-seq samples
- `SRTP000196[4-9]_raw_feature_bc_matrix.h5` - 10x raw count matrices for CITE-seq samples
- `SRTP000201[3-6]_filtered_feature_bc_matrix.h5` - Additional 10x filtered count matrices for CITE-seq samples
- `SRTP000201[3-6]_raw_feature_bc_matrix.h5` - Additional 10x raw count matrices for CITE-seq samples

**Multiome Raw Data (10x ATAC + Gene Expression):**
- `SRTP000197[0-9]_filtered_feature_bc_matrix.h5` - 10x filtered count matrices for Multiome samples
- `SRTP000200[0-2]_filtered_feature_bc_matrix.h5` - Additional 10x filtered count matrices for Multiome samples
- `SRTP000199[7-9]_atac_fragments.tsv.gz` - 10x ATAC fragment files containing chromatin accessibility data
- `SRTP000200[0-2]_atac_fragments.tsv.gz` - Additional 10x ATAC fragment files
- `SRTP000201[7-20]_atac_fragments.tsv.gz` - Additional 10x ATAC fragment files
- `SRTP000201[7-20]_filtered_feature_bc_matrix.h5` - Additional 10x filtered count matrices for Multiome samples

**Additional 10x Output Files:**
- `filtered_feature_bc_matrix.h5` - 10x filtered feature-barcode matrix
- `raw_feature_bc_matrix.h5` - 10x raw feature-barcode matrix
- `atac_fragments.tsv.gz` - 10x ATAC fragments file
- `atac_fragments.tsv.gz.tbi` - Index file for ATAC fragments
- `atac_peaks.bed` - 10x called ATAC peaks in BED format

### Processed Data Files

**Individual Sample Processed Files:**

*CITE-seq Processed:*
- `CITE_donor_[donor]_day_[day].h5mu` - CITE-seq data processed using 10x raw and filtered matrices. RNA modality: cell filtering (total counts 10,000-40,000, min 2,200 genes, <20% mitochondrial), gene filtering (min 20 cells), normalization (1M counts per cell), log transformation, highly variable gene detection (n=4,000), doublet removal using Scrublet, cell type annotation using marker gene scoring. Protein modality: DSB normalization using empty droplets for ambient correction and isotype controls for technical noise reduction.

*Multiome Processed:*
- `multiome_donor_[donor]_day_[day]_processed.h5mu` - Multiome data processed from 10x outputs. RNA modality: cell filtering (total counts 3,100-40,000, min 2,000 genes, <20% mitochondrial), gene filtering (min 20 cells), normalization (1M counts per cell), log transformation, highly variable gene detection (n=4,000), doublet removal using Scrublet, cell type annotation using marker gene scoring. ATAC modality: cell filtering (fragment counts 1,000-80,000, min 750 features, TSS enrichment >0.8, nucleosome signal <2.0), feature filtering (min 20 cells), TF-IDF normalization, latent semantic indexing (LSI), component filtering to remove depth-correlated components.

**Combined Processed Files:**

*CITE-seq Combined:*
- `citeseq_alldonors_alldays.h5ad` - Combined CITE-seq data across all donors and time points with harmonized processing parameters and batch-corrected analysis
- `citeseq_protein_counts.h5ad` - DSB-normalized protein expression data from CITE-seq experiments
- `citeseq_rna_counts.h5ad` - Processed RNA expression data from CITE-seq experiments with normalization and filtering applied

*Multiome Combined:*
- `multiome_alldonors_alldays.h5ad` - Combined multiome data across all donors and time points with harmonized processing for both RNA and ATAC modalities
- `multiome_atac_counts.h5ad` - Processed ATAC-seq data with TF-IDF normalization, LSI dimensionality reduction, and quality filtering
- `multiome_rna_counts.h5ad` - Processed RNA expression data from multiome experiments with standard scRNA-seq processing pipeline

*Integrated Analysis:*
- `rna_combined_with_velocity_and_refined_annotations.h5ad` - Integrated RNA expression data with RNA velocity analysis and refined cell type annotations using marker gene scoring and manual curation

### Checksums and Verification Files
- `*.md5` - MD5 checksums for data integrity verification
- `md5sum.txt` - Complete list of MD5 checksums for all files

## File Formats
- `.h5` - HDF5 format (10x Genomics standard output)
- `.h5ad` - AnnData HDF5 format (scanpy/Python ecosystem)
- `.h5mu` - MuData HDF5 format (multimodal data)
- `.tsv.gz` - Compressed tab-separated values
- `.bed` - Browser Extensible Data format for genomic regions
- `.tbi` - Tabix index files

## Sample Naming Convention
- Donor IDs: 13176, 27678, 31800, 32606
- Time points: day_2, day_3, day_4, day_7, day_10
- SRTP IDs correspond to individual sequencing library preparations