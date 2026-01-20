version 1.0

task list_images {
    input {
        Boolean? save_group_size
        Array[String] urls
        String? image_pattern
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
        from scallops.utils import _list_images_wdl

        save_group_size =  "~{save_group_size}" == 'true'
        image_pattern = "~{image_pattern}"
        urls = '~{sep="," urls}'.split(",")
        groupby = "~{sep=',' groupby}".split(",")
        subset = "~{sep=',' subset}".split(",")
        batch_size_str = "~{batch_size}"
        expected_cycles = "~{expected_cycles}"
        _list_images_wdl(image_pattern, urls, groupby, subset, batch_size_str, save_group_size, expected_cycles)
        CODE
    >>>

    output {
        Array[String] groups = read_lines('groups.txt')
        Array[String] t = read_lines('t.txt')
        Array[String] filtered_groupby = read_lines('groupby.txt')
        String groupby_pattern = read_lines('groupby_pattern.txt')[0]
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
