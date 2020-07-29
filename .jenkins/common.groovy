// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()
        
    project.compiler.compiler_path = platform.jenkinsLabel.contains('hip-clang') ? '/opt/rocm/bin/hipcc' : '/opt/rocm/bin/hcc'        
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ${buildTypeArg} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """
    
    platform.runCommand(this, command)
}


def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''

    def testCommand = "ctest${centos} --output-on-failure"
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                cd ${project.testDirectory}
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ${testCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project, jobName)
{
    def command

    if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/${project.testDirectory}
                make package
                rm -rf package && mkdir -p package
                mv *.rpm package/
                rpm -qlp package/*.rpm
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/${project.testDirectory}/package/*.rpm""")
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/${project.testDirectory}
                make package
                rm -rf package && mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
            """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/${project.testDirectory}/package/*.deb""")
    }
}

return this

