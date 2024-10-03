package org.examples

import org.gradle.accessors.dm.LibrariesForLibs
import org.gradle.kotlin.dsl.java
import org.gradle.kotlin.dsl.maven
import org.gradle.kotlin.dsl.repositories
import org.gradle.kotlin.dsl.the

plugins {
    java
}

val libs = the<LibrariesForLibs>()

repositories {
    mavenCentral()
    mavenLocal()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}
