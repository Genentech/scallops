version 1.0

import "utils.wdl" as utils
import "stitch_tasks.wdl" as tasks

workflow stitch_workflow {
    input {
        Array[String] urls
        String? image_pattern # e.g. Well{well}_Point{point}_Channel{channel}_Seq{seq}.nd2
        String output_directory
        Array[String] groupby = ["plate", "well", "t"]
        Int stitch_channel = 0
        Array[String]? subset

        Int? expected_images
        Float? stitch_max_shift
        String? stitch_blend
        String? stitch_radial_correction_k
        String? stitch_stage_positions
        Int? stitch_crop
        Float? stitch_min_overlap_fraction
        Array[Int]? stitch_output_channels
        Int? expected_cycles

        String? z_index
        String illumination_agg_method = "mean"
        String? rename # 2 column CSV file without a header that maps image id to new image id (e.g. 20240909_003053_292-3 -> 1-3)

        Boolean run_illumination_correction = true
        Int? stitch_cpu
        String? stitch_memory
        Int? illumination_correction_cpu
        String? illumination_correction_memory

        String stitch_disks = "local-disk 100 HDD"
        String illumination_correction_disks = "local-disk 100 HDD"

        String illumination_correction_suffix = "illumination_correction"
        String stitch_suffix = "stitch"

        # force
        Boolean? force_illumination_correction
        Boolean? force_stitch

        String docker = "563221710766.dkr.ecr.us-west-2.amazonaws.com/external/ctg/scallops:latest"

        Int preemptible = 0
        String zones = "us-west1-a us-west1-b us-west1-c"
        String aws_queue_arn = "arn:aws:batch:us-west-2:752311211819:job-queue/gred"
        Int max_retries = 0

    }

    String output_directory_stripped = sub(output_directory, "/+$", "")
    String stitch_output_directory = output_directory_stripped + "/" + stitch_suffix
    String illumination_correction_output_directory = output_directory_stripped + "/" + illumination_correction_suffix

    call utils.list_images {
        input:
            urls = urls,
            image_pattern = image_pattern,
            subset=subset,
            save_group_size=true,
            expected_cycles=expected_cycles,
            groupby=groupby,
            docker=docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries
    }
    Array[String] groups = list_images.groups

    Int group_size = list_images.group_size # assume <= 500 is 10x

    String illumination_correction_memory_default = if(group_size<=500) then "16 GiB" else "32 GiB"
    Int illumination_correction_cpu_default = if(group_size<=500) then 8 else 16

    String stitch_memory_default = if(group_size<=500) then "16 GiB" else "96 GiB"
    Int stitch_cpu_default = if(group_size<=500) then 8 else 48

    scatter (group in groups) {
        if(run_illumination_correction) {
            call tasks.illumination_correction {
                input:
                    images=urls,
                    image_pattern=image_pattern,
                    agg_method=illumination_agg_method,
                    z_index=z_index,
                    expected_images=expected_images,
                    subset = group,
                    groupby=list_images.filtered_groupby,
                    force=force_illumination_correction,
                    output_directory=illumination_correction_output_directory,
                    docker=docker,
                    zones = zones,
                    preemptible = preemptible,
                    aws_queue_arn = aws_queue_arn,
                    max_retries = max_retries,
                    disks = illumination_correction_disks,
                    memory = select_first([illumination_correction_memory, illumination_correction_memory_default]),
                    cpu = select_first([illumination_correction_cpu, illumination_correction_cpu_default])
            }
        }

        call tasks.stitch {
            input:
                images=urls,
                image_pattern=image_pattern,
                z_index=z_index,
                subset = group,
                groupby=list_images.filtered_groupby,
                expected_images=expected_images,
                output_directory=stitch_output_directory,
                force=force_stitch,
                rename=rename,
                output_channels=stitch_output_channels,
                channel=stitch_channel,
                flatfield_url=illumination_correction.output_url,
                radial_correction_k=stitch_radial_correction_k,
                stage_positions=stitch_stage_positions,
                crop=stitch_crop,
                min_overlap_fraction=stitch_min_overlap_fraction,
                max_shift=stitch_max_shift,
                blend=stitch_blend,
                docker=docker,
                zones = zones,
                preemptible = preemptible,
                aws_queue_arn = aws_queue_arn,
                max_retries = max_retries,
                disks = stitch_disks,
                memory = select_first([stitch_memory, stitch_memory_default]),
                cpu = select_first([stitch_cpu, stitch_cpu_default]) ,
        }
    }

    output {
        Array[String?] flatfield_images = illumination_correction.output_url
        Array[String] stitched_images = stitch.output_url

    }
}
