package org.examples

tasks {
    register("formatShell") {
        doLast {
            providers.exec {
                workingDir = projectDir
                commandLine(
                    "bash",
                    "-c",
                    "find tests serving -name '*.sh' | xargs shfmt -i 2 -w"
                )
            }
        }
    }
}
