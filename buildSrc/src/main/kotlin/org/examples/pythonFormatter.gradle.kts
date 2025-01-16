package org.examples

import org.gradle.kotlin.dsl.registering

tasks {
    register("formatPython") {
        doLast {
            providers.exec {
                workingDir = projectDir
                commandLine(
                    "bash",
                    "-c",
                    "find . -name '*.py' -not -path '*/.gradle/*' -not -path '*/build/*' -not -path '*/venv/*' -print0 | xargs -0 yapf --in-place"
                )
            }
        }
    }

    val verifyPython by registering {
        doFirst {
            try {
                providers.exec {
                    workingDir = projectDir
                    commandLine(
                        "bash",
                        "-c",
                        "find . -name '*.py' -not -path '*/.gradle/*' -not -path '*/build/*' -not -path '*/venv/*' -print0 | xargs -0 yapf -d"
                    )
                }
            } catch (e: Exception) {
                throw GradleException(
                    "Repo is improperly formatted, please run ./gradlew formatPython, and recommit",
                    e
                )
            }
        }
    }
}
