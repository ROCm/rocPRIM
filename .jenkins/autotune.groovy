#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI = 
{
    nodeDetails, jobName->

    def prj = new rocProject('rocPRIM', 'PreCheckin')
    prj.testDirectory = 'scripts/autotune'

    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    def commonGroovy

    boolean formatCheck = false

    def testCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runTestCommand(platform, project, auxiliary.gfxTargetParser() + 
            '\n./autotune_usage_script.sh $(sed -e "s/gfx([0-9a-f]+).*/\1/" <<< $gpu_arch)')
    }


    buildProject(prj, formatCheck, nodes.dockerArray, null, testCommand, null)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi":[pipelineTriggers([cron('0 1 * * 0')])], 
                        "compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])],
                        "rocm-docker":[]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900'],centos7:['gfx906'],centos8:['gfx906'],sles15sp1:['gfx908']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList, 'rocPRIM')

    propertyList.each 
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }
    
    jobNameList.each 
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu16:['gfx906']], urlJobName)
        }
    }
}
