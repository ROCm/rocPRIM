// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean isCodeCovOn=false)
{
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String codeCovArg = isCodeCovOn ? '-DBUILD_CODE_COVERAGE=ON' : ''
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} --toolchain=toolchain-linux.cmake ${buildTypeArg} ${amdgpuTargets} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ${codeCovArg} ../..
                make -j\$(nproc)
                """

    platform.runCommand(this, command)
}


def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def testCommand = "ctest --output-on-failure "
    def testCommandExcludeRegex = ''
    def testCommandExclude = "--exclude-regex \"${testCommandExcludeRegex}\""
    def hmmExcludeRegex = ''
    def hmmTestCommandExclude = "--exclude-regex \"${hmmExcludeRegex}\""
    def hmmTestCommand = ''
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        echo("HMM TESTS DISABLED")
        /*hmmTestCommand = """
                            export HSA_XNACK=1
                            export ROCPRIM_USE_HMM=1
                            ${testCommand} ${hmmTestCommandExclude}
                         """*/
    }
    echo(env.JOB_NAME)
    if (env.JOB_NAME.contains('bleeding-edge'))
    {
        testCommand = ''
        testCommandExclude = ''
        hmmTestCommand = ''
        echo("TESTS DISABLED")
    }
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                cd ${project.testDirectory}
                ${testCommand} ${testCommandExclude}
                if (( \$? != 0 )); then
                    exit 1
                fi
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
}

def runCodecovTestCommand(platform, project)
{
    String gfilter = "*"
    String dirmode = "release"

    def testCommand = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${dirmode}
                ls
                cat Makefile
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG make coverage_cleanup coverage GTEST_FILTER=${gfilter}
            """

    platform.runCommand(this, testCommand)

    this.publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/build/${dirmode}/lcoverage",
                reportFiles: "index.html",
                reportName: "Code coverage report",
                reportTitles: "Code coverage report"])

    if (this.env.BRANCH_NAME ==~ /PR-\d+/)
    {
        def commentBody = "${this.env.BUILD_URL}../Code_20coverage_20report/"
        
        this.writeFile(file: 'comment.py', text: this.libraryResource("com/amd/scripts/comment.py"))

        this.withCredentials([this.string(credentialsId: 'ROCmMathLibrariesBot-PAT', variable: 'GH_AUTH')])
        {
            def commentCommand = """
            python comment.py -u ROCmMathLibrariesBot -o ROCm -r ${project.paths.project_name} -n ${this.env.CHANGE_ID} -b ${commentBody}
            """
            platform.runCommand(this, commentCommand)
        }
    }
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this
