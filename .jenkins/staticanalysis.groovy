#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()
    
    def command = """#!/usr/bin/env bash
            set -ex
            cd ${project.paths.project_build_prefix}/docs
            doxygen
            """
    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        throw e
    }
    
    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/html",
                reportFiles: 'index.html',
                reportName: 'Documentation',
                reportTitles: 'Documentation'])
}


def runCI =
{
    nodeDetails, jobName->
    
    def prj = new rocProject('rocPRIM', 'StaticAnalysis')

    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = true
    boolean staticAnalysis = true
    
    def commonGroovy
     
    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 9 * * 6')])]))
    
    stage(urlJobName) {
        runCI([ubuntu18:['cpu']], urlJobName)
    }
}
