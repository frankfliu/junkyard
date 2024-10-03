package org.examples

tasks {
    register("formatShell") {
        doLast {
            project.exec {
                commandLine(
                    "bash",
                    "-c",
                    "find tests serving -name '*.sh' | xargs shfmt -i 2 -w"
                )
            }
        }
    }
}
