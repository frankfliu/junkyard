plugins {
    `kotlin-dsl`
}

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation("com.google.googlejavaformat:google-java-format:1.27.0")
    implementation("com.github.spotbugs.snom:spotbugs-gradle-plugin:6.1.11")
    implementation(files(libs.javaClass.superclass.protectionDomain.codeSource.location))
}
