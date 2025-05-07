plugins {
    org.examples.javaProject
}

group = "org.example"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.itextpdf:layout:9.1.0") {
        exclude("com.itextpdf","bouncy-castle-connector")
    }
    implementation(libs.slf4j.simple)

    testImplementation(libs.testng) {
        exclude(group = "junit", module = "junit")
    }
}
