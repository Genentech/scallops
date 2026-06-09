version 1.1

task list_images {
    input {
        Boolean? save_group_size
        Array[String] urls
        String? image_pattern
        String? reference_time
        Array[String] groupby
        Array[String]? subset
        Int? expected_cycles
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

        save_group_size =  "~{save_group_size}" == 'true'
        image_pattern = "~{image_pattern}"
        reference_time = "~{reference_time}"
        urls = '~{sep="," urls}'.split(",")
        groupby = "~{sep=',' groupby}".split(",")
        subset = "~{sep=',' subset}".split(",")
        batch_size_str = "~{batch_size}"
        expected_cycles = "~{expected_cycles}"
        _list_images_wdl(image_pattern, urls, groupby, reference_time, subset, batch_size_str, save_group_size, expected_cycles)
        CODE
    >>>

    output {
        Array[String] subsets = read_lines('subsets.txt')
        Array[String] subset_with_reference_times = read_lines('subsets_with_t.txt')
        Array[String] t = read_lines('t.txt')
        String groupby_pattern = read_lines('groupby_pattern.txt')[0] # e.g. {plate}-{well}
        String groupby_pattern_with_reference_t = read_lines('groupby_pattern_with_reference_t.txt')[0] # e.g. {plate}-{well}-IF
        Array[String] filtered_groupby_with_t = read_lines('groupby_with_t.txt') # e.g. [plate, well, t]
        Array[String] filtered_groupby = read_lines('groupby.txt') # e.g. [plate, well]

        Int group_size = read_int('group_size.txt')
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
