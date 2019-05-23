name := "spark-xkmeans"

organization := "org.tupol"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.0"
val onlineStatsVersion = "0.0.2"


// ------------------------------
// DEPENDENCIES AND RESOLVERS

resolvers += "Sonatype OSS Releases" at "https://oss.sonatype.org/content/repositories/releases"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

lazy val providedDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion force(),
  "org.apache.spark" %% "spark-sql" % sparkVersion force(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion force()
)

libraryDependencies ++= providedDependencies.map(_ % "provided")

libraryDependencies ++= Seq(
  "org.tupol" %% "online-stats" % onlineStatsVersion,
  "org.scalacheck" %% "scalacheck" % "1.12.5" % "test",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test"
)

// ------------------------------
// TESTING
parallelExecution in Test := false

fork in Test := true

publishArtifact in Test := true

// ------------------------------
// RUNNING

// Make sure that provided dependencies are added to classpath when running in sbt
run in Compile <<= Defaults.runTask(fullClasspath in Compile,
  mainClass in(Compile, run),
  runner in(Compile, run))

fork in run := true

// ------------------------------
// PUBLISHING
isSnapshot := version.value.trim.endsWith("SNAPSHOT")

useGpg := true

// Nexus (see https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html)
publishTo := {
  val repo = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at repo + "content/repositories/snapshots")
  else
    Some("releases" at repo + "service/local/staging/deploy/maven2")
}

publishArtifact in Test := true

publishMavenStyle := true

pomIncludeRepository := { _ => false }

licenses := Seq("MIT-style" -> url("https://opensource.org/licenses/MIT"))

homepage := Some(url("https://github.com/tupol/spark-xkmeans"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/tupol/spark-xkmeans.git"),
    "scm:git@github.com:tupol/spark-xkmeans.git"
  )
)

developers := List(
  Developer(
    id    = "tupol",
    name  = "Tupol",
    email = "tupol.github@gmail.com",
    url   = url("https://github.com/tupol")
  )
)

releasePublishArtifactsAction := PgpKeys.publishSigned.value
import sbtrelease.ReleasePlugin.autoImport.ReleaseTransformations._
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,          // performs the initial git checks
  tagRelease,
  releaseStepCommand(s"""sonatypeOpen "${organization.value}" "${name.value} v${version.value}""""),
  releaseStepCommand("publishSigned"),
  releaseStepCommand("sonatypeRelease"),
  setNextVersion,
  commitNextVersion,
  pushChanges                     // also checks that an upstream branch is properly configured
)

// ------------------------------

