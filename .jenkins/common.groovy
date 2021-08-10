// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ${buildTypeArg} ${amdgpuTargets} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """

    platform.runCommand(this, command)
}


def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String installPackage = ""
    if (platform.jenkinsLabel.contains("centos") || platform.jenkinsLabel.contains("sles"))
    {
        installPackage = "sudo rpm -i rocprim*.rpm"
    }
    else 
    {
        installPackage = "sudo dpkg -i rocprim*.deb"
    }
    String runTests = ""
    String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''

    def testCommand = "ctest${centos} --output-on-failure "
    def hmmTestCommand = ''
    def testCommandsExclude = ["device_scan"]
    def hmmExcludeRegex = /(rocprim.device_merge|rocprim.device_scan|rocprim.device_run_length_encode|rocprim.device_segmented_radix_sort|rocprim.device_partition|rocprim.device_radix_sort)/
    def hmmTestCommandExclude = "--exclude-regex \"${hmmExcludeRegex}\""
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        hmmTestCommand = """
                            export HSA_XNACK=1
                            export ROCPRIM_USE_HMM=1
                            ${testCommand} ${hmmTestCommandExclude}
                         """
    }
    String rocprim_tests = "/opt/rocm/rocprim/bin/*"
    if (testCommandsExclude.size() > 0)
    {
        String excludeNameCondition = testCommandsExclude.collect("-name 'test_${it}'").join(' \\| ')
        rocprim_tests = "\$(find /opt/rocm/rocprim/bin/* \\! \\( ${excludeNameCondition} \\))"
    }
    runTests = """
                pushd ${project.paths.project_build_prefix}
                mv build build_BAK
                for test in ${rocprim_tests}; do
                    \$test
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                done
                mv build_BAK build
                popd
    """
    def command = """#!/usr/bin/env bash
                set -x
                pushd ${project.paths.project_build_prefix}/build/release/package
                ${installPackage}
                popd
                ${runTests}
                cd ${project.testDirectory}
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release", true)

    platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this

