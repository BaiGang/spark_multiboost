addSbtPlugin("com.github.mpeltonen" % "sbt-idea" % "1.4.0")

addSbtPlugin("com.typesafe.sbteclipse" % "sbteclipse-plugin" % "2.1.1")

resolvers += Classpaths.typesafeResolver

addSbtPlugin("com.github.retronym" % "sbt-onejar" % "0.8")

addSbtPlugin("org.scalastyle" %% "scalastyle-sbt-plugin" % "0.3.2")

addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.9.2")

// adding support for source code formatting using Scalariform
addSbtPlugin("com.typesafe.sbt" % "sbt-scalariform" % "1.0.1")

