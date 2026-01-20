version 1.0

import "utils.wdl" as utils
import "ops_tasks.wdl" as tasks

workflow ops_workflow {
    input {
        String? iss_url
        String iss_image_pattern = "{plate}-{well}-{t}"

        String? phenotype_url
        String phenotype_image_pattern =  "{plate}-{well}-{t}"

        Array[String] groupby = ["plate", "well"]

        String output_directory

        # t to align phenotyping rounds to e.g. "IF"
        String? reference_phenotype_time

        # features
        Array[String]? phenotype_cell_features
        Array[String]? phenotype_nuclei_features
        Array[String]? phenotype_cytosol_features
        String? features_extra_arguments # Single string with extra arguments to scallops features cli

        Int? features_cell_min_area
        Int? features_nuclei_min_area
        Int? features_cytosol_min_area
        Int? features_cell_max_area
        Int? features_nuclei_max_area
        Int? features_cytosol_max_area

        Array[Int] phenotype_cyto_channel # indices after registration for cell segmentation
        Int phenotype_dapi_channel # index after registration for segmentation and pheno to iss registration
        Int? phenotype_dapi_channel_before_registration # for pheno to pheno registration

        Int? iss_dapi_channel # ISS to ISS and pheno to ISS registration

        String? iss_registration_extra_arguments # Extra arguments in scallops registration elastix cli for ISS
        String? pheno_to_iss_registration_extra_arguments
        String? pheno_registration_extra_arguments
        Boolean? register_across_channels

        # spot detect
        Int? iss_expected_cycles
        Array[Int] iss_channels =  [1,2,3,4]
        Array[Float] spot_detection_sigma_log = [1]
        Int? spot_detection_max_filter_width
        Int? spot_detection_peak_neighborhood_size
        String? spot_detection_extra_arguments

        # reads
        String reads_labels # "nuclei" or "cell"
        String? barcodes
        String? barcode_column
        Boolean? all_reads
        Int? reads_expand_labels_distance
        Int? reads_mismatches

        Boolean? reads_crosstalk_correction_by_t
        String? reads_bases
        String? reads_threshold_peaks
        String? reads_threshold_peaks_crosstalk
        String? reads_extra_arguments

        String model_dir = "s3://bigdipir-ctg-s3/models/"

        # nuclei segment
        String? nuclei_segmentation
        String? nuclei_segmentation_extra_arguments

        # cell segment
        String? cell_segmentation_method
        String? segment_cell_threshold
        Float? segment_cell_threshold_correction_factor
        String? cell_segmentation_extra_arguments

        Boolean mark_stitch_boundary_cells = true

        # merge
        String? merge_extra_arguments

        # force
        Boolean run_spot_detect = true
        Boolean run_nuclei_segmentation = true
        Boolean run_cell_segmentation = true
        Boolean force_segment_nuclei = false
        Boolean force_segment_cell  = false
        Boolean force_spot_detect  = false
        Boolean force_reads  = false
        Boolean force_merge  = false
        Boolean force_register_iss  = false
        Boolean force_register_pheno_to_iss  = false
        Boolean force_register_pheno_to_pheno  = false
        Boolean force_features = false
        Boolean force_find_objects = false
        Boolean force_register_pheno_to_iss_qc = false

        # general options
        Array[String]? subset
        Int? batch_size # for processing multiple images in one batch

        # resources
        Int segment_nuclei_cpu = 128
        String segment_nuclei_memory = "256 GiB"
        String segment_nuclei_disks = "local-disk 20 HDD"

        Int segment_cell_cpu = 32
        String segment_cell_memory = "256 GiB"
        String segment_cell_disks = "local-disk 20 HDD"

        Int register_iss_cpu = 32
        String register_iss_disks = "local-disk 20 HDD"
        String register_iss_memory = "64 GiB"

        Int register_pheno_to_iss_cpu = 64
        String register_pheno_to_iss_memory = "256 GiB"
        String register_pheno_to_iss_disks = "local-disk 20 HDD"

        Int register_pheno_to_pheno_cpu = 64
        String register_pheno_to_pheno_memory = "256 GiB"
        String register_pheno_to_pheno_disks = "local-disk 20 HDD"

        Int features_cpu = 96
        String features_memory = "384 GiB"
        String features_disks = "local-disk 200 HDD"

        Int find_objects_cpu = 16
        String find_objects_memory = "32 GiB"
        String find_objects_disks = "local-disk 200 HDD"

        Int spot_detect_cpu = 32
        String spot_detect_memory = "256 GiB"
        String spot_detect_disks = "local-disk 20 HDD"

        Int reads_cpu = 16
        String reads_memory = "128 GiB"
        String reads_disks = "local-disk 20 HDD"

        String register_pheno_to_iss_qc_memory = "96 GiB"
        Int register_pheno_to_iss_qc_cpu = 48
        String register_pheno_to_iss_qc_disks = "local-disk 200 HDD"

        Int merge_cpu = 32
        String merge_memory = "256 GiB"
        String merge_disks = "local-disk 20 HDD"

        Int cell_intersects_boundary_cpu = 16
        String cell_intersects_boundary_memory = "32 GiB"
        String cell_intersects_boundary_disks = "local-disk 200 HDD"


        String docker = "563221710766.dkr.ecr.us-west-2.amazonaws.com/external/ctg/scallops:latest"

        Int preemptible = 0
        String zones = "us-west1-a us-west1-b us-west1-c"
        String aws_queue_arn = "arn:aws:batch:us-west-2:752311211819:job-queue/gred"
        Int max_retries = 0

        String segment_suffix = "segment.zarr"
        String register_iss_suffix = "iss-registered-t0.zarr"
        String register_iss_transforms_suffix = "iss-transforms-t0"
        String register_pheno_to_iss_suffix = "pheno-to-iss-registered.zarr"
        String register_pheno_to_iss_transforms_suffix = "pheno-to-iss-transforms"
        String nuclei_objects_suffix = "objects-nuclei"
        String cell_objects_suffix = "objects-cell"
        String cytosol_objects_suffix = "objects-cytosol"
        String nuclei_features_suffix = "features-nuclei"
        String cell_features_suffix = "features-cell"
        String cytosol_features_suffix = "features-cytosol"
        String register_pheno_to_pheno_suffix = "pheno-registered.zarr"
        String register_pheno_to_pheno_transform_suffix = "pheno-to-pheno-transforms"
        String register_pheno_to_iss_qc_suffix = "pheno-to-iss-qc"
        String spot_detect_suffix = "spot-detect.zarr"
        String reads_suffix = "reads"
        String merge_suffix = "merge"
        String cell_intersects_boundary_suffix = "intersects-boundary"
        String cell_intersects_boundary_non_reference_t_suffix = "intersects-boundary-t"
    }


    String output_stripped = sub(output_directory, "/+$", "") + "/"
    String segment_directory = output_stripped + segment_suffix
    String register_iss_t0_directory = output_stripped + register_iss_suffix
    String register_iss_t0_transforms_directory = output_stripped + register_iss_transforms_suffix
    String register_pheno_to_iss_directory = output_stripped + register_pheno_to_iss_suffix
    String register_pheno_to_iss_transforms_directory = output_stripped + register_pheno_to_iss_transforms_suffix
    String nuclei_features_directory = output_stripped + nuclei_features_suffix
    String cell_features_directory = output_stripped + cell_features_suffix
    String cytosol_features_directory = output_stripped + cytosol_features_suffix
    String nuclei_objects_directory =  output_stripped + nuclei_objects_suffix
    String cell_objects_directory =  output_stripped + cell_objects_suffix
    String cytosol_objects_directory =  output_stripped + cytosol_objects_suffix
    String register_pheno_to_pheno_directory = output_stripped + register_pheno_to_pheno_suffix
    String register_pheno_to_pheno_transform_directory = output_stripped + register_pheno_to_pheno_transform_suffix
    String spot_detect_directory = output_stripped + spot_detect_suffix
    String reads_directory = output_stripped + reads_suffix
    String merge_directory = output_stripped + merge_suffix
    String register_pheno_to_iss_qc_directory = output_stripped + register_pheno_to_iss_qc_suffix
    String cell_intersects_boundary_directory = output_stripped + cell_intersects_boundary_suffix
    String cell_intersects_boundary_directory_non_reference_t = output_stripped + cell_intersects_boundary_non_reference_t_suffix

    Boolean iss_url_supplied = defined(iss_url)
    Boolean pheno_url_supplied = defined(phenotype_url)

    call utils.list_images {
        input:
            urls = [select_first([phenotype_url, iss_url])],
            image_pattern = if pheno_url_supplied then phenotype_image_pattern else iss_image_pattern,
            batch_size=batch_size,
            groupby=groupby,
            subset=subset,
            docker=docker,
            zones = zones,
            preemptible = preemptible,
            aws_queue_arn = aws_queue_arn,
            max_retries = max_retries
    }
    String image_pattern_after_registration = list_images.groupby_pattern
    Array[String] groups = list_images.groups
    Array[String] times = list_images.t
    scatter (group in groups) {
        if(pheno_url_supplied) {
            if(length(times)>1) {
                call tasks.register_elastix as register_pheno_to_pheno {
                    input:
                        moving=select_all([phenotype_url]),
                        moving_label=phenotype_url, # transform stitch masks
                        moving_channel=phenotype_dapi_channel_before_registration, # DAPI index in each round
                        moving_image_pattern=phenotype_image_pattern,
                        reference_time=reference_phenotype_time,
                        extra_arguments=pheno_registration_extra_arguments,
                        unroll_channels=true,
                        register_across_channels=register_across_channels,
                        groupby=groupby,
                        moving_output_directory=register_pheno_to_pheno_directory,
                        label_output_directory=register_pheno_to_pheno_directory,
                        transform_output_directory=register_pheno_to_pheno_transform_directory,
                        subset = group,
                        force = force_register_pheno_to_pheno,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = register_pheno_to_pheno_disks,
                        memory = register_pheno_to_pheno_memory,
                        cpu = register_pheno_to_pheno_cpu,
                        max_retries = max_retries
                }
            }
            String register_pheno_to_pheno_output_url = select_first([register_pheno_to_pheno.moving_output_url, phenotype_url])
            String register_pheno_to_pheno_image_pattern = if(length(times)>1) then image_pattern_after_registration else phenotype_image_pattern

            if(run_nuclei_segmentation) {
                call tasks.segment_nuclei {
                    input:
                        images = register_pheno_to_pheno_output_url,
                        image_pattern = register_pheno_to_pheno_image_pattern,
                        method = nuclei_segmentation,
                        groupby=groupby,
                        dapi_channel = phenotype_dapi_channel,
                        output_directory=segment_directory,
                        model_dir=model_dir,
                        subset = group,
                        extra_arguments=nuclei_segmentation_extra_arguments,
                        force = force_segment_nuclei,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = segment_nuclei_disks,
                        memory = segment_nuclei_memory,
                        cpu = segment_nuclei_cpu,
                        max_retries = max_retries
                }
            }
            if(run_cell_segmentation) {
                call tasks.segment_cell {
                    input:
                        images = register_pheno_to_pheno_output_url,
                        image_pattern = register_pheno_to_pheno_image_pattern,
                        method = cell_segmentation_method,
                        groupby=groupby,
                        dapi_channel = phenotype_dapi_channel,
                        cyto_channel=phenotype_cyto_channel,
                        nuclei_label=select_first([segment_nuclei.output_url]),
                        threshold=segment_cell_threshold,
                        threshold_correction_factor = segment_cell_threshold_correction_factor,
                        output_directory=segment_directory,
                        model_dir=model_dir,
                        subset = group,
                        extra_arguments=cell_segmentation_extra_arguments,
                        force = force_segment_cell,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = segment_cell_disks,
                        memory = segment_cell_memory,
                        cpu = segment_cell_cpu,
                        max_retries = max_retries
                }
                call tasks.find_objects as find_objects_cell {
                    input:
                        labels= segment_cell.output_url,
                        label_pattern=image_pattern_after_registration,
                        suffix="cell",
                        output_directory=cell_objects_directory,
                        subset = group,
                        force = force_find_objects,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = find_objects_disks,
                        memory = find_objects_memory,
                        cpu = find_objects_cpu,
                        max_retries = max_retries
                }

                # determine whether cells intersect stitch boundary
                # using stitch mask as image
                if(mark_stitch_boundary_cells) {
                    String t0 = if (length(times)>0) then times[0] else ""
                    String reference_phenotype_time_ = select_first([reference_phenotype_time, t0])
                    String output_prefix = if (reference_phenotype_time_!="") then "-" else ""
                    String phenotype_url_stripped = if (pheno_url_supplied) then sub(select_first([phenotype_url]), "/+$", "")  else ""
                    call tasks.intersects_boundary as cell_intersects_boundary {
                        # reference time mask is not transformed
                        # use mask from stitch output
                        input:
                            labels=segment_cell.output_url,
                            images=phenotype_url_stripped + '/labels/',
                            image_pattern=image_pattern_after_registration + output_prefix + reference_phenotype_time_ + '-mask',
                            output_directory=cell_intersects_boundary_directory,
                            label_type='cell',
                            objects=find_objects_cell.output_url,
                            groupby=groupby,
                            subset = group,
                            force = force_segment_cell,
                            docker=docker,
                            zones = zones,
                            preemptible = preemptible,
                            aws_queue_arn = aws_queue_arn,
                            disks = cell_intersects_boundary_disks,
                            memory = cell_intersects_boundary_memory,
                            cpu = cell_intersects_boundary_cpu,
                            max_retries = max_retries
                    }
                    if (length(times)>1) {
                        call tasks.intersects_boundary as cell_intersects_boundary_t {
                            # non-reference time masks are transformed
                            # use masks from registration output
                            input:
                                labels= segment_cell.output_url,
                                images=register_pheno_to_pheno.moving_output_url + '/labels/',
                                image_pattern=phenotype_image_pattern + '-mask',
                                output_directory=cell_intersects_boundary_directory_non_reference_t,
                                label_type='cell',
                                objects=find_objects_cell.output_url,
                                subset = group,
                                groupby=groupby,
                                force = force_segment_cell,
                                docker=docker,
                                zones = zones,
                                preemptible = preemptible,
                                aws_queue_arn = aws_queue_arn,
                                disks = cell_intersects_boundary_disks,
                                memory = cell_intersects_boundary_memory,
                                cpu = cell_intersects_boundary_cpu,
                                max_retries = max_retries
                        }
                    }
                }
            }
        }

        if(iss_url_supplied) {
            call tasks.register_elastix as register_iss_t0 {
                input:
                    moving=[select_first([iss_url])],
                    moving_image_pattern=iss_image_pattern,
                    moving_channel=iss_dapi_channel,
                    groupby=groupby,
                    moving_output_directory=register_iss_t0_directory,
                    transform_output_directory=register_iss_t0_transforms_directory,
                    register_across_channels=register_across_channels,
                    extra_arguments=iss_registration_extra_arguments,
                    subset = group,
                    force = force_register_iss,
                    docker=docker,
                    zones = zones,
                    preemptible = preemptible,
                    aws_queue_arn = aws_queue_arn,
                    disks = register_iss_disks,
                    memory = register_iss_memory,
                    cpu = register_iss_cpu,
                    max_retries = max_retries
            }
        }

        if(iss_url_supplied && pheno_url_supplied) {
            call tasks.register_elastix as register_pheno_to_iss {
                input:
                    fixed=select_first([iss_url]),
                    fixed_channel=iss_dapi_channel,
                    moving_label=segment_cell.output_url,
                    moving=select_all([register_pheno_to_pheno_output_url]),
                    moving_image_pattern=register_pheno_to_pheno_image_pattern,
                    fixed_image_pattern=iss_image_pattern,
                    moving_channel=phenotype_dapi_channel,
                    output_aligned_channels_only=true,
                    register_across_channels=register_across_channels,
                    moving_output_directory=register_pheno_to_iss_directory,
                    label_output_directory=register_pheno_to_iss_directory,
                    transform_output_directory=register_pheno_to_iss_transforms_directory,
                    subset = group,
                    groupby=groupby,
                    extra_arguments=pheno_to_iss_registration_extra_arguments,
                    force = force_register_pheno_to_iss,
                    docker=docker,
                    zones = zones,
                    preemptible = preemptible,
                    aws_queue_arn = aws_queue_arn,
                    disks = register_pheno_to_iss_disks,
                    memory = register_pheno_to_iss_memory,
                    cpu = register_pheno_to_iss_cpu,
                    max_retries = max_retries
            }
            if(run_nuclei_segmentation) {
                call tasks.find_objects as find_objects_nuclei {
                    input:
                        labels=segment_nuclei.output_url,
                        label_pattern=image_pattern_after_registration,
                        suffix="nuclei",
                        output_directory=nuclei_objects_directory,
                        subset = group,
                        force = force_find_objects,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = find_objects_disks,
                        memory = find_objects_memory,
                        cpu = find_objects_cpu,
                        max_retries = max_retries
                }

                call tasks.register_qc as register_pheno_to_iss_qc {
                    input:
                        images=select_first([register_iss_t0.moving_output_url]),
                        image_pattern=image_pattern_after_registration,
                        stacked_images=register_pheno_to_iss.moving_output_url,
                        stacked_image_pattern=image_pattern_after_registration,
                        image_channel=iss_dapi_channel,
                        stacked_image_channel=0,
                        label_type='nuclei',
                        output_directory=register_pheno_to_iss_qc_directory,
                        labels=register_pheno_to_iss.label_output_url,
                        subset = group,
                        groupby=groupby,
                        force = force_register_pheno_to_iss_qc,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = register_pheno_to_iss_qc_disks,
                        memory = register_pheno_to_iss_qc_memory,
                        cpu = register_pheno_to_iss_qc_cpu,
                        max_retries = max_retries
                }
            }

        }

        if (defined(phenotype_nuclei_features)) {

            Array[String] phenotype_nuclei_features_ = select_first([phenotype_nuclei_features])
            # cromwell hack
            Int features_nuclei_min_area_ = select_first([features_nuclei_min_area, -1])
            Int features_nuclei_max_area_ = select_first([features_nuclei_max_area, -1])
            scatter (index in range(length(phenotype_nuclei_features_))) {

                call tasks.features as features_nuclei {
                    input:
                        images = select_first([register_pheno_to_pheno_output_url]),
                        image_pattern=register_pheno_to_pheno_image_pattern,
                        nuclei_features = phenotype_nuclei_features_[index],
                        nuclei_min_area = features_nuclei_min_area_,
                        nuclei_max_area = features_nuclei_max_area_,
                        features_extra_arguments=features_extra_arguments,
                        labels= segment_cell.output_url,
                        objects=find_objects_nuclei.output_url,
                        model_dir=model_dir,
                        groupby=groupby,
                        output_directory=nuclei_features_directory + '-' + index,
                        subset = group,
                        force = force_features,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = features_disks,
                        memory = features_memory,
                        cpu = features_cpu,
                        max_retries = max_retries
                }
            }
        }

        if (defined(phenotype_cell_features)) {

            Array[String] phenotype_cell_features_ = select_first([phenotype_cell_features])
            # cromwell hack
            Int features_cell_min_area_ = select_first([features_cell_min_area, -1])
            Int features_cell_max_area_ = select_first([features_cell_max_area, -1])
            scatter (index in range(length(phenotype_cell_features_))) {
                call tasks.features as features_cell {
                    input:
                        images = select_first([register_pheno_to_pheno_output_url]),
                        image_pattern=register_pheno_to_pheno_image_pattern,
                        cell_features = phenotype_cell_features_[index],
                        cell_min_area = features_cell_min_area_,
                        cell_max_area = features_cell_max_area_,
                        features_extra_arguments=features_extra_arguments,
                        labels= segment_cell.output_url,
                        objects=find_objects_cell.output_url,
                        model_dir=model_dir,
                        groupby=groupby,
                        output_directory=cell_features_directory + '-' + index,
                        subset = group,
                        force = force_features,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = features_disks,
                        memory = features_memory,
                        cpu = features_cpu,
                        max_retries = max_retries
                }
            }
        }
         if (run_cell_segmentation) {
            call tasks.find_objects as find_objects_cytosol {
                input:
                    labels=segment_cell.output_url,
                    label_pattern=image_pattern_after_registration,
                    suffix="cytosol",
                    output_directory=cytosol_objects_directory,
                    subset = group,
                    force = force_find_objects,
                    docker=docker,
                    zones = zones,
                    preemptible = preemptible,
                    aws_queue_arn = aws_queue_arn,
                    disks = find_objects_disks,
                    memory = find_objects_memory,
                    cpu = find_objects_cpu,
                    max_retries = max_retries
            }
         }
        if (defined(phenotype_cytosol_features)) {

            Array[String] phenotype_cytosol_features_ = select_first([phenotype_cytosol_features])
            # cromwell hack
            Int features_cytosol_min_area_ = select_first([features_cytosol_min_area, -1])
            Int features_cytosol_max_area_ = select_first([features_cytosol_max_area, -1])
            scatter (index in range(length(phenotype_cytosol_features_))) {
                call tasks.features as features_cytosol {
                    input:
                        images = select_first([register_pheno_to_pheno_output_url]),
                        image_pattern=register_pheno_to_pheno_image_pattern,
                        cytosol_features = phenotype_cytosol_features_[index],
                        cytosol_min_area = features_cytosol_min_area_,
                        cytosol_max_area = features_cytosol_max_area_,
                        labels = segment_cell.output_url,
                        objects = find_objects_cytosol.output_url,
                        features_extra_arguments=features_extra_arguments,
                        model_dir=model_dir,
                        groupby=groupby,
                        output_directory=cytosol_features_directory + '-' + index,
                        subset = group,
                        force = force_features,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = features_disks,
                        memory = features_memory,
                        cpu = features_cpu,
                        max_retries = max_retries
                }
            }
        }

        if(iss_url_supplied && run_spot_detect) {
            call tasks.spot_detect {
                input:
                    images=select_first([register_iss_t0.moving_output_url]),
                    image_pattern=image_pattern_after_registration,
                    iss_channels=iss_channels,
                    sigma_log=spot_detection_sigma_log,
                    max_filter_width=spot_detection_max_filter_width,
                    peak_neighborhood_size=spot_detection_peak_neighborhood_size,
                    expected_cycles=iss_expected_cycles,
                    output_directory=spot_detect_directory,
                    subset = group,
                    groupby=groupby,
                    extra_arguments=spot_detection_extra_arguments,
                    force = force_spot_detect,
                    docker=docker,
                    zones = zones,
                    preemptible = preemptible,
                    aws_queue_arn = aws_queue_arn,
                    disks = spot_detect_disks,
                    memory = spot_detect_memory,
                    cpu = spot_detect_cpu,
                    max_retries = max_retries
            }
            if(defined(barcodes)) {
                call tasks.reads {
                    input:
                        spots=spot_detect.output_url,
                        labels=select_first([register_pheno_to_iss.label_output_url]),
                        barcodes=select_first([barcodes]),
                        barcode_column=barcode_column,
                        output_directory=reads_directory,
                        bases=reads_bases,
                        all_reads=all_reads,
                        crosstalk_correction_by_t=reads_crosstalk_correction_by_t,
                        threshold_peaks=reads_threshold_peaks,
                        expand_labels_distance=reads_expand_labels_distance,
                        label_name=reads_labels,
                        mismatches=reads_mismatches,
                        threshold_peaks_crosstalk=reads_threshold_peaks_crosstalk,
                        subset = group,
                        extra_arguments=reads_extra_arguments,
                        force = force_reads,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = reads_disks,
                        memory = reads_memory,
                        cpu = reads_cpu,
                        max_retries = max_retries
                }
            }
            if (defined(barcodes)) {

                call tasks.merge {
                    input:
                        iss_reads=select_first([reads.output_url]) + '/labels',
                        phenotypes_nuclei=features_nuclei.output_url,
                        phenotypes_cell=features_cell.output_url,
                        phenotypes_cytosol=features_cytosol.output_url,
                        objects_nuclei=find_objects_nuclei.output_url,
                        objects_cell=find_objects_cell.output_url,
                        objects_cytosol=find_objects_cytosol.output_url,
                        cell_intersects_boundary=cell_intersects_boundary.output_url,
                        cell_intersects_boundary_t=cell_intersects_boundary_t.output_url,
                        register_pheno_to_iss_qc=register_pheno_to_iss_qc.output_url,
                        barcodes=select_first([barcodes]),
                        barcode_column=barcode_column,
                        output_directory=merge_directory,
                        subset = group,
                        extra_arguments=merge_extra_arguments,
                        force = force_merge,
                        docker=docker,
                        zones = zones,
                        preemptible = preemptible,
                        aws_queue_arn = aws_queue_arn,
                        disks = merge_disks,
                        memory = merge_memory,
                        cpu = merge_cpu,
                        max_retries = max_retries
                }
            }
        }
    }
    output {
        Array[String?] segment_nuclei_output_url = segment_nuclei.output_url
        Array[String?] segment_cell_output_url = segment_cell.output_url
        Array[String?] register_iss_t0_output_url = register_iss_t0.moving_output_url
        Array[String?] register_pheno_to_iss_output_url = register_pheno_to_iss.moving_output_url
        Array[String?] register_pheno_to_iss_qc_output_url = register_pheno_to_iss_qc.output_url
        Array[String?] register_pheno_to_pheno_moving_output_url = register_pheno_to_pheno.moving_output_url
        Array[String?] spot_detect_output_url = spot_detect.output_url
        Array[String?] reads_output_url = reads.output_url
        Array[String?] find_objects_nuclei_output_url = find_objects_nuclei.output_url
        Array[String?] find_objects_cell_output_url = find_objects_cell.output_url
        Array[String?] find_objects_cytosol_output_url = find_objects_cytosol.output_url
        Array[Array[String]?] features_nuclei_output_url = features_nuclei.output_url
        Array[Array[String]?] features_cell_output_url = features_cell.output_url
        Array[Array[String]?] features_cytosol_output_url = features_cytosol.output_url
        Array[String?] merge_output_url = merge.output_url
        Array[String] list_images_groups = list_images.groups

    }
}
