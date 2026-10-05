version 1.0

task filter {
    input {
        Array[String] inputs
        String output_path
        Array[String] by = []
        Int? n_features
        Float? min_feature_variance
        Float? max_feature_variance
        Float? max_cell_fraction_not_finite

        Boolean? force
        String? logging
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

    Boolean has_by = length(by) > 0

    command <<<
        set -ex

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map filter \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        ~{true="--by" false="" has_by} ~{sep=" " by} \
        ~{"--n-features " + n_features} \
        ~{"--min-feature-variance " + min_feature_variance} \
        ~{"--max-feature-variance " + max_feature_variance} \
        ~{"--max-cell-fraction-not-finite " + max_cell_fraction_not_finite} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task normalize {
    input {
        Array[String] inputs
        String output_path
        Array[String] by = []
        String? method
        Int? neighbors
        Float? max_value
        Array[String] centroid_columns = []
        String? reference_query
        String? mad_scale_factor
        Boolean? robust
        Boolean? no_centering
        Boolean? no_scaling
        Boolean? force
        String? logging
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

    Boolean has_by = length(by) > 0
    Boolean has_centroid_columns = length(centroid_columns) > 0

    command <<<
        set -ex

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map normalize \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        ~{true="--by" false="" has_by} ~{sep=" " by} \
        ~{"--method " + method} \
        ~{"--neighbors " + neighbors} \
        ~{"--max-value " + max_value} \
        ~{true="--centroid-columns" false="" has_centroid_columns} ~{sep=" " centroid_columns} \
        ~{if defined(reference_query) then '--reference-query "' + reference_query + '"' else ''} \
        ~{"--mad-scale-factor " + mad_scale_factor} \
        ~{true="--robust" false="" robust} \
        ~{true="--no-centering" false="" no_centering} \
        ~{true="--no-scaling" false="" no_scaling} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task pca {
    input {
        Array[String] inputs
        String output_path
        Int? components
        Int? batch_size
        Boolean? whiten
        String? reference_query
        Boolean? force
        String? logging
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

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map pca \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        ~{"--components " + components} \
        ~{"--batch-size " + batch_size} \
        ~{true="--whiten" false="" whiten} \
        ~{if defined(reference_query) then '--reference-query "' + reference_query + '"' else ''} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task tvn {
    input {
        Array[String] inputs
        String output_path
        String reference_query
        Array[String] by = []
        Int? pca_batch_size
        Boolean? force
        String? logging
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

    Boolean has_by = length(by) > 0

    command <<<
        set -ex

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map tvn \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        --reference-query "~{reference_query}" \
        ~{true="--by" false="" has_by} ~{sep=" " by} \
        ~{"--pca-batch-size " + pca_batch_size} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task aggregate {
    input {
        Array[String] inputs
        String output_path
        Array[String] by
        String? center_reference_query
        Boolean? force
        String? logging
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

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map aggregate \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        --by ~{sep=" " by} \
        ~{if defined(center_reference_query) then '--center-reference-query "' + center_reference_query + '"' else ''} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task similarity_matrix {
    input {
        Array[String] inputs
        String output_path
        Boolean? force
        String? logging
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

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map similarity-matrix \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task recall {
    input {
        Array[String] inputs
        String output_path
        Array[String] ground_truth_corum
        Array[Float] threshold = []
        Boolean? force
        String? logging
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

    Boolean has_threshold = length(threshold) > 0

    command <<<
        set -ex

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map recall \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        --ground-truth-corum "~{sep='" "' ground_truth_corum}" \
        ~{true="--threshold" false="" has_threshold} ~{sep=" " threshold} \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}

task enrichment {
    input {
        Array[String] inputs
        String output_path
        Array[String] sets
        Boolean? force
        String? logging
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

        ~{if defined(logging) then 'export SCALLOPS_LOGGING="' + logging + '"' else ''}

        scallops pert-map enrichment \
        --input "~{sep='" "' inputs}" \
        --output "~{output_path}" \
        --set "~{sep='" "' sets}" \
        ~{true="--force" false="" force} \
        ~{if defined(extra_arguments) then extra_arguments else ''}
    >>>

    output {
        String output_url = "~{output_path}"
    }

    runtime {
        docker: docker
        disks: disks
        zones: zones
        memory: memory
        cpu : cpu
        preemptible: preemptible
        queueArn: aws_queue_arn
        maxRetries : max_retries
    }
}
