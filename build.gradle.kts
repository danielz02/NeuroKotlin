plugins {
    kotlin("jvm") version "1.3.71"
    application
    id("org.jlleitschuh.gradle.ktlint") version "9.2.1"
}

group = "ch.danielz"
version = "1.0-SNAPSHOT"

repositories {
    maven("https://dl.bintray.com/kotlin/kotlin-numpy")
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("org.jetbrains:kotlin-numpy:0.1.4")
    implementation("org.nield:kotlin-statistics:1.2.1")
    testImplementation("io.kotest:kotest-runner-junit5-jvm:4.0.4") // for kotest framework
    testImplementation("io.kotest:kotest-assertions-core-jvm:4.0.4") // for kotest core jvm assertions
    testImplementation("io.kotest:kotest-property-jvm:4.0.4") // for kotest property test
}

tasks {
    compileKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
    compileTestKotlin {
        kotlinOptions.jvmTarget = "1.8"
    }
}

tasks.withType<Test> {
    useJUnitPlatform()
}
