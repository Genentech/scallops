version 1.0

task illumination_correction {
    input {
        Array[String] images
        String subset
        String? z_index
        String agg_method
        Int? expected_images
        String? image_pattern
        Array[String] groupby
        String output_directory
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
    #Array[String] images_quoted = quote(images)

    command <<<
        set -e

        python <<CODE
        from subprocess import check_call

        force = "~{force}"
        expected_images = "~{expected_images}"
        image_pattern = "~{image_pattern}"
        z_index = "~{z_index}"
        cmd = ["scallops", "illum-corr", "agg"]
        cmd += ["--images"]
        cmd += "~{sep=',' images}".split(",")
        if image_pattern != "":
            cmd += ["--image-pattern", image_pattern]
        cmd += ["--groupby"]
        cmd += "~{sep=',' groupby}".split(",")
        cmd += ["--subset", "~{subset}"]
        cmd += ["--output-image-format", "tiff"]
        cmd += ["-o",  "~{output_directory}/"]
        cmd += ["--agg-method", "~{agg_method}"]
        if force=="true":
            cmd.append("--force")
        if expected_images != "":
            cmd.append("--expected-images")
            cmd.append(expected_images)
        if z_index != "":
            cmd.append("--z-index")
            cmd.append(z_index)
        print(' '.join(cmd))
        check_call(cmd)
        CODE
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

task stitch {
    input {
        Array[String] images
        String? flatfield_url
        Int? expected_images
        String? rename
        Int channel
        String? z_index
        String subset
        Array[String] groupby
        Array[Int]? output_channels
        String? image_pattern
        String output_directory

        String? radial_correction_k
        String? stage_positions
        Int? crop
        Float? min_overlap_fraction
        Float? max_shift

        Boolean? force
        String? blend
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

        python <<CODE
        import subprocess

        z_index = "~{z_index}"
        image_output = "~{output_directory}" + "/stitch.zarr"
        report_output = "~{output_directory}" + "/stitch-report"
        force = "~{force}" == "true"
        rename = "~{rename}"
        expected_images = "~{expected_images}"
        radial_correction_k = "~{radial_correction_k}"
        stage_positions = "~{ stage_positions}"
        crop = "~{crop}"
        min_overlap_fraction = "~{min_overlap_fraction}"
        subset = "~{subset}"
        blend = "~{blend}"
        max_shift = "~{max_shift}"
        image_pattern = "~{image_pattern}"
        flatfield_url = "~{flatfield_url}"
        groupby = '~{sep="," groupby}'.split(",")
        images = '~{sep="," images}'.split(",")
        output_channels = [s.strip() for s in '~{sep="," output_channels}'.split(",") if s.strip()!='']
        channel = "~{channel}"
        z_index_url = None
        if flatfield_url != "":
            ffp = "~{flatfield_url}/" + "-".join(["{" + g + "}" for g in groupby]) + ".ome.tiff"
            z_index_url = "~{flatfield_url}/" + "-".join(["{" + g + "}" for g in groupby]) + "-zindex.parquet"
        cmd = ["scallops", "stitch"]
        cmd += ["--images"] + images
        if image_pattern != "":
            cmd += ["--image-pattern", image_pattern]
        cmd += ["--groupby"] + groupby
        cmd += ["--subset", subset]
        cmd += ["-c", channel]
        if max_shift != "":
            cmd += ["--max-shift", max_shift]
        if radial_correction_k != "":
            cmd += ["--radial-correction-k", radial_correction_k]
        if crop != "":
            cmd += ["--crop", crop]
        if blend != "":
            cmd += ["--blend", blend]
        if stage_positions != "":
            cmd += ["--stage-positions", stage_positions]
        if min_overlap_fraction != "":
            cmd += ["--min-overlap-fraction", min_overlap_fraction]
        if flatfield_url != "":
            cmd += ["--ffp", ffp]
        cmd += ["--image-output", image_output]
        cmd += ["--report-output", report_output]

        if force:
            cmd.append("--force")
        if rename != "":
            cmd += ["--rename", rename]
        if expected_images != "":
            cmd.append("--expected-images")
            cmd.append(expected_images)
        if len(output_channels) > 0:
             cmd += ["--output-channels"] + output_channels
        if z_index != "":
            cmd.append("--z-index")
            if z_index=="focus" and z_index_url is not None:
                cmd.append(z_index_url)
            else:
                cmd.append(z_index)
        print(' '.join(cmd))
        subprocess.check_call(cmd)

        CODE
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
