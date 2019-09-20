#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocprimCI:
{

    def rocprim = new rocProject('rocPRIM')

    def nodes = new dockerNodes(['gfx803 && ubuntu && hip-clang', 'gfx900 && centos && hip-clang', 'gfx906 && ubuntu && hip-clang'], rocprim)

    boolean formatCheck = false
     
    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                """
        
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    sudo ctest --output-on-failure
                """

        platform.runCommand(this, command)
    }

    def packageCommand = null

    buildProject(rocprim, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
