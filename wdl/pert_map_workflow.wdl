version 1.0

import "pert_map_tasks.wdl" as tasks

# Perturbation map pipeline:
#   filter -> normalize -> pca -> tvn -> aggregate -> similarity-matrix -> {recall, enrichment}
workflow pert_map_workflow {
    input {
        Array[String] inputs # one or more zarr/h5ad/Parquet paths or patterns (e.g. s3://foo/*.zarr)
        String output_directory

        Array[String] by = ["plate", "well"] # grouping for filter / normalize / tvn
        Array[String] aggregate_by = ["gene_symbol"]



        # filter
        Int? filter_n_features
        Float? filter_min_feature_variance
        Float? filter_max_feature_variance
        Float? filter_max_cell_fraction_not_finite
        String? filter_extra_arguments

        # normalize
        String normalize_method ="local-zscore"
        Int normalize_neighbors = 100
        Float? normalize_max_value
        Array[String] normalize_centroid_columns = ["Nuclei_AreaShape_Center_Y", "Nuclei_AreaShape_Center_X"]
        String? normalize_extra_arguments

        # pca
        Int pca_components = 128
        Int? pca_batch_size
        Boolean? pca_whiten
        String? pca_reference_query
        String? pca_extra_arguments

        # tvn
        String tvn_reference_query
        String? tvn_extra_arguments

        # aggregate
        String? aggregate_center_reference_query
        String? aggregate_extra_arguments

        # similarity matrix
        String? similarity_matrix_extra_arguments

        # recall - skipped unless ground truth is supplied
        Array[String] recall_ground_truth_corum = ["xxx"]
        Array[Float] recall_threshold = [0.99, 0.95, 0.01, 0.05]
        String? recall_extra_arguments

        # enrichment - skipped unless at least one GMT is supplied
        Array[String] enrichment_sets = ["xxx"]
        String? enrichment_extra_arguments

        # force
        Boolean force_filter = false
        Boolean force_normalize = false
        Boolean force_pca = false
        Boolean force_tvn = false
        Boolean force_aggregate = false
        Boolean force_similarity_matrix = false
        Boolean force_recall = false
        Boolean force_enrichment = false

        String? logging # value for SCALLOPS_LOGGING (e.g. "debug")


        # output file names, written under output_directory
        String filter_output_name = "filter.zarr"
        String normalize_output_name = "normalize.zarr"
        String pca_output_name = "pca.zarr"
        String tvn_output_name = "tvn.zarr"
        String aggregate_output_name = "agg.zarr"
        String similarity_matrix_output_name = "sim.zarr"
        String recall_output_name = "recall.parquet"
        String enrichment_output_name = "enrichment.parquet"

        # resources
        Int filter_cpu = 32
        String filter_memory = "256 GiB"
        String filter_disks = "local-disk 100 HDD"

        Int normalize_cpu = 32
        String normalize_memory = "256 GiB"
        String normalize_disks = "local-disk 100 HDD"

        Int pca_cpu = 32
        String pca_memory = "256 GiB"
        String pca_disks = "local-disk 100 HDD"

        Int tvn_cpu = 32
        String tvn_memory = "256 GiB"
        String tvn_disks = "local-disk 100 HDD"

        Int aggregate_cpu = 32
        String aggregate_memory = "256 GiB"
        String aggregate_disks = "local-disk 100 HDD"

        Int similarity_matrix_cpu = 8
        String similarity_matrix_memory = "32 GiB"
        String similarity_matrix_disks = "local-disk 50 HDD"

        Int recall_cpu = 4
        String recall_memory = "16 GiB"
        String recall_disks = "local-disk 50 HDD"

        Int enrichment_cpu = 4
        String enrichment_memory = "16 GiB"
        String enrichment_disks = "local-disk 50 HDD"

        String docker

        Int preemptible = 0
        String zones = "us-west1-a us-west1-b us-west1-c"
        String aws_queue_arn = ""
        Int max_retries = 0
    }

    String output_directory_stripped = sub(output_directory, "/+$", "")

    call tasks.filter {
        input:
            inputs = inputs,
            output_path = output_directory_stripped + "/" + filter_output_name,
            by = by,
            n_features = filter_n_features,
            min_feature_variance = filter_min_feature_variance,
            max_feature_variance = filter_max_feature_variance,
            max_cell_fraction_not_finite = filter_max_cell_fraction_not_finite,
            force = force_filter,
            logging = logging,
            extra_arguments = filter_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = filter_cpu,
            memory = filter_memory,
            disks = filter_disks
    }

    call tasks.normalize {
        input:
            inputs = [filter.output_url],
            output_path = output_directory_stripped + "/" + normalize_output_name,
            by = by,
            method = normalize_method,
            neighbors = normalize_neighbors,
            max_value = normalize_max_value,
            centroid_columns = normalize_centroid_columns,

            force = force_normalize,
            logging = logging,
            extra_arguments = normalize_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = normalize_cpu,
            memory = normalize_memory,
            disks = normalize_disks
    }

    call tasks.pca {
        input:
            inputs = [normalize.output_url],
            output_path = output_directory_stripped + "/" + pca_output_name,
            components = pca_components,
            batch_size = pca_batch_size,
            whiten = pca_whiten,
            reference_query = pca_reference_query,
            force = force_pca,
            logging = logging,
            extra_arguments = pca_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = pca_cpu,
            memory = pca_memory,
            disks = pca_disks
    }

    call tasks.tvn {
        input:
            inputs = [pca.output_url],
            output_path = output_directory_stripped + "/" + tvn_output_name,
            reference_query = tvn_reference_query,
            by = by,
            force = force_tvn,
            logging = logging,
            extra_arguments = tvn_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = tvn_cpu,
            memory = tvn_memory,
            disks = tvn_disks
    }

    call tasks.aggregate {
        input:
            inputs = [tvn.output_url],
            output_path = output_directory_stripped + "/" + aggregate_output_name,
            by = aggregate_by,
            center_reference_query = aggregate_center_reference_query,
            force = force_aggregate,
            logging = logging,
            extra_arguments = aggregate_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = aggregate_cpu,
            memory = aggregate_memory,
            disks = aggregate_disks
    }

    call tasks.similarity_matrix {
        input:
            inputs = [aggregate.output_url],
            output_path = output_directory_stripped + "/" + similarity_matrix_output_name,
            force = force_similarity_matrix,
            logging = logging,
            extra_arguments = similarity_matrix_extra_arguments,
            docker = docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries,
            cpu = similarity_matrix_cpu,
            memory = similarity_matrix_memory,
            disks = similarity_matrix_disks
    }

    if (length(recall_ground_truth_corum) > 0) {
        call tasks.recall {
            input:
                inputs = [similarity_matrix.output_url],
                output_path = output_directory_stripped + "/" + recall_output_name,
                ground_truth_corum = recall_ground_truth_corum,
                threshold = recall_threshold,
                force = force_recall,
                logging = logging,
                extra_arguments = recall_extra_arguments,
                docker = docker,
                zones = zones,
                preemptible = preemptible,
                aws_queue_arn = aws_queue_arn,
                max_retries = max_retries,
                cpu = recall_cpu,
                memory = recall_memory,
                disks = recall_disks
        }
    }

    if (length(enrichment_sets) > 0) {
        call tasks.enrichment {
            input:
                inputs = [similarity_matrix.output_url],
                output_path = output_directory_stripped + "/" + enrichment_output_name,
                sets = enrichment_sets,
                force = force_enrichment,
                logging = logging,
                extra_arguments = enrichment_extra_arguments,
                docker = docker,
                zones = zones,
                preemptible = preemptible,
                aws_queue_arn = aws_queue_arn,
                max_retries = max_retries,
                cpu = enrichment_cpu,
                memory = enrichment_memory,
                disks = enrichment_disks
        }
    }

    output {
        String filtered = filter.output_url
        String normalized = normalize.output_url
        String pca_output = pca.output_url
        String tvn_output = tvn.output_url
        String aggregated = aggregate.output_url
        String similarity = similarity_matrix.output_url
        String? recall_output = recall.output_url
        String? enrichment_output = enrichment.output_url
    }
}
