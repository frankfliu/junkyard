package org.examples

import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.gradle.kotlin.dsl.attributes
import org.gradle.kotlin.dsl.`java-library`
import org.gradle.kotlin.dsl.systemProperties

plugins {
    id("org.examples.javaBase")
    `java-library`
    id("org.examples.javaFormatter")
    id("org.examples.check")
}

tasks {
    compileJava {
        options.apply {
            release = 11
            encoding = "UTF-8"
            compilerArgs = listOf("-Xlint:all,-options,-static", "-Werror")
        }
    }
    compileTestJava {
        options.apply {
            release = 11
            encoding = "UTF-8"
            compilerArgs = listOf("-proc:none", "-Xlint:all,-options,-static,-removal", "-Werror")
        }
    }
    javadoc {
        options {
            this as StandardJavadocDocletOptions // https://github.com/gradle/gradle/issues/7038
            addStringOption("Xdoclint:none", "-quiet")
        }
    }
    test {
        // tensorflow mobilenet and resnet require more cpu memory
        maxHeapSize = "4096m"

        useTestNG {
            //suiteXmlFiles = listOf(File(rootDir, "testng.xml")) //This is how to add custom testng.xml
        }

        testLogging {
            showStandardStreams = true
            events("passed", "skipped", "failed", "standardOut", "standardError")
            exceptionFormat = TestExceptionFormat.FULL
        }

        environment("MODEL_SERVER_HOME", "${project.projectDir}")

        jvmArgs("--add-opens", "java.base/jdk.internal.loader=ALL-UNNAMED")
        for (prop in System.getProperties().iterator()) {
            val key = prop.key.toString()
            if (key.startsWith("ai.djl.")) {
                systemProperty(key, prop.value)
            }
        }
        systemProperties(
            "org.slf4j.simpleLogger.defaultLogLevel" to "debug",
            "org.slf4j.simpleLogger.log.org.mortbay.log" to "warn",
            "org.slf4j.simpleLogger.log.org.testng" to "info",
        )
        if (gradle.startParameter.isOffline)
            systemProperty("ai.djl.offline", "true")
    }
}
