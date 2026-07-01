version 1.0

task list_images {
    input {

        Array[String]? urls1
        String? image_pattern1
        String? reference_time1
        Int? n_cycles1

        Array[String]? urls2
        String? image_pattern2
        String? reference_time2
        Int? n_cycles2

        Array[String] groupby
        Array[String]? subset

        Boolean? save_group_size
        Int? batch_size
        String docker
        String zones
        Int preemptible
        String aws_queue_arn
        Int max_retries
    }

    Int cpu = 1
    String disks = "local-disk 10 HDD"
    String memory = "4 GiB"

    command <<<
        set -e
        python <<CODE
        from scallops.cli.util import _list_images_wdl


        urls1 = '~{sep="," urls1}'.split(",")
        image_pattern1 = "~{image_pattern1}"
        reference_time1 = "~{reference_time1}"
        n_cycles1 = "~{n_cycles1}"

        urls2 = '~{sep="," urls2}'.split(",")
        image_pattern2 = "~{image_pattern2}"
        reference_time2 = "~{reference_time2}"
        n_cycles2 = "~{n_cycles2}"

        groupby = "~{sep=',' groupby}".split(",")
        subset = "~{sep=',' subset}".split(",")
        batch_size = "~{batch_size}"
        save_group_size =  "~{save_group_size}" == 'true'

        _list_images_wdl(image_pattern1=image_pattern1, urls1=urls1, n_cycles1=n_cycles1, reference_time1=reference_time1,
        image_pattern2=image_pattern2, urls2=urls2, n_cycles2=n_cycles2, reference_time2=reference_time2, subset=subset,
        batch_size=batch_size, save_group_size=save_group_size, groupby=groupby)
        CODE
    >>>

    output {
        Array[String] subsets = read_lines('subsets.txt')
        String groupby_pattern = read_lines('groupby_pattern.txt')[0] # e.g. {plate}-{well}
        Array[String] groupby_array = read_lines('groupby_array.txt') # e.g. ["plate", "well"]

        Array[String] groupby_array_with_time_1 = read_lines('groupby_array_with_time_1.txt') # e.g. ["plate", "well", "t"]
        Array[String] groupby_array_with_time_2 = read_lines('groupby_array_with_time_2.txt') # e.g. ["plate", "well", "t"]

        Int group_size_1 = read_int('group_size_1.txt')
        Int group_size_2 = read_int('group_size_2.txt')

        Array[String] times_1 = read_lines('times_1.txt')
        Array[String] times_2 = read_lines('times_2.txt')

        String reference_time_1 = read_lines('reference_time_1.txt')[0]
        String reference_time_2 = read_lines('reference_time_2.txt')[0]



    }

    meta {
        volatile: true
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
