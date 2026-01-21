version 1.0

task segment_nuclei {
    input {
        String? method
        String images
        String? image_pattern
        Array[String] groupby
        Int? dapi_channel
        String output_directory
        String subset
        Boolean? force
        String model_dir
        String? extra_arguments
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        export SCALLOPS_MODEL_DIR="~{model_dir}"

        scallops segment nuclei \
        --images "~{images}" \
        ~{"--method " + method} \
        --groupby ~{sep=" " groupby} \
        ~{if defined(image_pattern) then '--image-pattern "' + image_pattern + '"' else ''} \
        ~{'--dapi-channel ' + dapi_channel} \
        --output "~{output_directory}" \
        --subset ~{subset} \
        ~{if defined(extra_arguments) then extra_arguments else ''} \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task segment_cell {
    input {
        String? method
        String images
        String? image_pattern
        Array[String] groupby
        Int? dapi_channel
        Array[Int] cyto_channel
        Int? chunks
        String? nuclei_label
        String? threshold
        Float? threshold_correction_factor
        String output_directory
        String model_dir
        String subset
        String? extra_arguments
        Boolean? force

        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        export SCALLOPS_MODEL_DIR="~{model_dir}"

        scallops segment cell \
        --images "~{images}" \
        --groupby ~{sep=" " groupby} \
        ~{if defined(image_pattern) then '--image-pattern "' + image_pattern + '"' else ''} \
        ~{'--dapi-channel ' + dapi_channel} \
        --cyto-channel ~{sep=" " cyto_channel} \
        ~{"--nuclei-label " + nuclei_label} \
        ~{"--method " + method} \
        --output "~{output_directory}" \
        --subset ~{subset} \
        ~{'--threshold ' + threshold} \
        ~{'--threshold-correction-factor ' + threshold_correction_factor} \
        ~{'--chunks ' + chunks} \
        ~{if defined(extra_arguments) then extra_arguments else ''} \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task register_elastix {
    input {
        Array[String] moving
        Array[String] groupby
        String? moving_image_pattern
        String? fixed_image_pattern
        Array[String]? sort
        Int? moving_channel
        Boolean? output_aligned_channels_only
        Boolean? unroll_channels
        String? fixed
        String? reference_time
        Int? fixed_channel
        Boolean? register_across_channels
        String transform_output_directory
        String? label_output_directory
        String? moving_output_directory
        String subset
        Boolean? force
        String? moving_label
        String? extra_arguments

        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        scallops registration elastix \
        --groupby ~{sep=" " groupby} \
        ~{if defined(moving_image_pattern) then '--moving-image-pattern "' + moving_image_pattern + '"' else ''} \
        ~{if defined(sort) then '--sort ' else ''} \
        ~{sep=" " sort} \
        ~{"--moving-channel " + moving_channel} \
        ~{if defined(moving_output_directory) && moving_output_directory !="" then '--moving-output "' + moving_output_directory + '"' else ''} \
        --moving ~{sep=" " moving} \
        ~{if defined(moving_label) then '--moving-label "' + moving_label + '"' else ''} \
        ~{if defined(fixed) then '--fixed "' + fixed + '"' else ''} \
        ~{"--fixed-channel " + fixed_channel} \
        ~{if defined(fixed_image_pattern) then '--fixed-image-pattern "' + fixed_image_pattern + '"' else ''} \
        --transform-output "~{transform_output_directory}" \
        --subset ~{subset} \
        ~{if defined(label_output_directory) then '--label-output "' + label_output_directory + '"' else ''} \
        ~{true="--unroll-channels" false="" unroll_channels} \
        ~{if defined(reference_time) then '--time "' + reference_time + '"' else ''} \
        ~{true="--force" false="" force} \
        ~{true="--align-across-channels" false="" register_across_channels} \
        ~{true="--output-aligned-channels-only" false="" output_aligned_channels_only} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String label_output_url = select_first([label_output_directory, ""])
        String moving_output_url = select_first([moving_output_directory, ""])

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task register_qc {
    input {
        String images
        String? image_pattern
        String label_type
        String labels
        String? stacked_images
        String? stacked_image_pattern
        Int? image_channel
        Int? stacked_image_channel
        String subset
        String output_directory
        Array[String] groupby
        Boolean? force

        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }
    Int image_channel_ = select_first([image_channel, 0])
    Int stacked_image_channel_ = select_first([stacked_image_channel, 0])
    command <<<
        set -ex

        if [[ "$SCALLOPS_TEST" != "1" ]]; then
            ulimit -n 100000
        fi


        scallops features \
        --features-~{label_type} "correlationpearsonbox_~{image_channel_}_s~{stacked_image_channel_}" \
        --labels "~{labels}" \
        --groupby ~{sep=" " groupby} \
        --subset ~{subset} \
        --output "~{output_directory}" \
        --images "~{images}" \
        --stack-images "~{stacked_images}" \
        ~{'--image-pattern ' + image_pattern} \
        ~{'--stack-image-pattern ' + stacked_image_pattern} \
        ~{true="--force" false="" force} \
        --features-plot "Nuclei_Correlation_PearsonBox_DAPI-ISS_DAPI-PHENO" \
        --channel-rename '{"~{image_channel_}":"ISS","s~{stacked_image_channel_}":"PHENO"}'
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task intersects_boundary {
    input {
        String images
        String? image_pattern
        String label_type
        String labels
        String subset
        String? objects
        String output_directory
        Array[String] groupby
        Boolean? force

        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        if [[ "$SCALLOPS_TEST" != "1" ]]; then
            ulimit -n 100000
        fi

        scallops features \
        --features-~{label_type} "intersects-boundary_0" \
        --labels "~{labels}" \
        --groupby ~{sep=" " groupby} \
        --subset ~{subset} \
        --output "~{output_directory}" \
        --images "~{images}" \
        --objects "~{objects}" \
        --no-normalize \
        ~{'--image-pattern ' + image_pattern} \
        ~{true="--force" false="" force}

    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task find_objects {
    input {
        String? labels
        String subset
        Boolean? force
        String? label_pattern
        String suffix
        String output_directory
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        if [[ "$SCALLOPS_TEST" != "1" ]]; then
            ulimit -n 100000
        fi

        scallops find-objects \
        --labels "~{labels}" \
        --subset ~{subset} \
        ~{"--label-pattern " + label_pattern} \
        --label-suffix ~{suffix} \
        --output "~{output_directory}" \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task features {
    input {
        String? nuclei_features
        String? cell_features
        String? cytosol_features
        Int? nuclei_min_area
        Int? cell_min_area
        Int? cytosol_min_area
        Int? nuclei_max_area
        Int? cell_max_area
        Int? cytosol_max_area
        String? features_extra_arguments
        String? model_dir
        String? labels
        String? objects
        String images
        String subset
        Boolean? force
        String? image_pattern
        Array[String] groupby
        String output_directory
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        export SCALLOPS_MODEL_DIR="~{model_dir}"

        if [[ "$SCALLOPS_TEST" != "1" ]]; then
            ulimit -n 100000
        fi

        scallops features \
        ~{"--features-nuclei " + nuclei_features} \
        ~{"--features-cell " + cell_features} \
        ~{"--features-cytosol " + cytosol_features} \
        ~{if defined(nuclei_min_area) && select_first([nuclei_min_area])>0 then '--nuclei-min-area ' + nuclei_min_area else ''} \
        ~{if defined(cell_min_area) && select_first([cell_min_area])>0 then '--cell-min-area ' + cell_min_area else ''} \
        ~{if defined(cytosol_min_area) && select_first([cytosol_min_area])>0 then '--cytosol-min-area ' + cytosol_min_area else ''} \
        ~{if defined(nuclei_max_area) && select_first([nuclei_max_area])>0 then '--nuclei-max-area ' + nuclei_max_area else ''} \
        ~{if defined(cell_max_area) && select_first([cell_max_area])>0 then '--cell-max-area ' + cell_max_area else ''} \
        ~{if defined(cytosol_max_area) && select_first([cytosol_max_area])>0 then '--cytosol-max-area ' + cytosol_max_area else ''} \
        ~{if defined(features_extra_arguments) then features_extra_arguments else ''} \
        --labels "~{labels}" \
        ~{"--objects " +objects} \
        --subset ~{subset} \
        ~{"--image-pattern " + image_pattern} \
        --groupby ~{sep=" " groupby} \
        --output "~{output_directory}" \
        --images "~{images}" \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task spot_detect {
    input {
        String images
        String image_pattern
        String subset
        Boolean? force
        Array[String] groupby
        Array[Int] iss_channels
        Array[Float] sigma_log
        Int? max_filter_width
        Int? peak_neighborhood_size
        Int? expected_cycles
        Int? chunks
        String output_directory
        String? extra_arguments
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        scallops pooled-sbs spot-detect \
        --output "~{output_directory}" \
        --image-pattern "~{image_pattern}" \
        --images "~{images}" \
        ~{"--max-filter-width " + max_filter_width} \
        ~{"--peak-neighborhood-size " + peak_neighborhood_size} \
        ~{"--expected-cycles " + expected_cycles} \
        --channel ~{sep=" " iss_channels} \
        --sigma-log ~{sep=" " sigma_log} \
        ~{"--chunks " + chunks} \
        --groupby ~{sep=" " groupby} \
        ~{if defined(extra_arguments) then extra_arguments else ''} \
        --subset ~{subset} \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task reads {
    input {
        String spots
        String subset
        Boolean? force
        String labels
        String? bases
        Boolean? all_reads
        Boolean? crosstalk_correction_by_t
        Int? mismatches
        Int? expand_labels_distance
        String? threshold_peaks
        String? threshold_peaks_crosstalk
        String output_directory
        String barcodes
        String? barcode_column
        String label_name
        String? extra_arguments
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -ex

        scallops pooled-sbs reads \
        --spots "~{spots}" \
        --labels "~{labels}" \
        --label-name "~{label_name}" \
        ~{"--bases " + bases} \
        ~{"--threshold-peaks " + threshold_peaks} \
        ~{"--threshold-peaks-crosstalk " + threshold_peaks_crosstalk} \
        ~{"--mismatches " + mismatches} \
        ~{"--expand-labels-distance " + expand_labels_distance} \
        ~{"--barcode-col " + barcode_column} \
        --output "~{output_directory}" \
        --subset ~{subset} \
        --barcodes "~{barcodes}" \
        ~{true="--all-reads" false="" all_reads} \
        ~{true="--crosstalk-correction-by-t" false="" crosstalk_correction_by_t} \
        ~{if defined(extra_arguments) then extra_arguments else ''} \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task merge {
    input {
        String iss_reads
        Array[String]? phenotypes_nuclei
        Array[String]? phenotypes_cell
        Array[String]? phenotypes_cytosol
        String? objects_nuclei
        String? objects_cell
        String? register_pheno_to_iss_qc
        String? objects_cytosol
        String? cell_intersects_boundary
        String? cell_intersects_boundary_t

        String output_directory
        String barcodes
        String? barcode_column
        String subset
        String? extra_arguments
        Boolean? force

        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int cpu
        String disks
        String memory
        Int max_retries
    }

    command <<<
        set -e


        scallops pooled-sbs merge \
        --sbs "~{iss_reads}" \
        --output "~{output_directory}" \
        --barcodes "~{barcodes}" \
        --phenotype \
        ~{sep=" " phenotypes_nuclei} \
        ~{sep=" " phenotypes_cell} \
        ~{sep=" " phenotypes_cytosol} \
        ~{objects_nuclei} \
        ~{objects_cell} \
        ~{objects_cytosol} \
        ~{cell_intersects_boundary} \
        ~{cell_intersects_boundary_t} \
        ~{register_pheno_to_iss_qc} \
        --subset ~{subset} \
        ~{"--barcode-col " + barcode_column} \
        ~{if defined(extra_arguments) then extra_arguments else ''} \
        ~{true="--force" false="" force}
    >>>

    output {
        String output_url = "~{output_directory}"

    }

    runtime {
        docker:docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}
