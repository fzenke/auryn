dnl Copyright 2014 Ankur Sinha - ankursinha@fedoraproject.org
dnl Configure.ac file for Auryn

dnl defile library version
dnl library versioning has nothing to do with package release versioning - remember
dnl haven't managed to get these to work in src/Makefile.am yet. Gotta try some more tricks
LIB_MAJOR=0
LIB_MINOR=0
LIB_MICRO=0
AC_SUBST(LIB_MAJOR)
AC_SUBST(LIB_MINOR)
AC_SUBST(LIB_MICRO)


AC_INIT([auryn], [0.4.1], [friedemann.zenke@epfl.ch])
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE(dist-bzip2 check-news)

dnl cpp it is
AC_LANG([C++])

dnl Initialise libtool
LT_INIT([shared static])

AC_CONFIG_SRCDIR([src/System.cpp])

AM_MAINTAINER_MODE

dnl check for c++ compiler
AC_PROG_CXX
dnl check for install command
AC_PROG_INSTALL
dnl check for libtool
AC_PROG_LIBTOOL
dnl MPI
AC_ARG_ENABLE([mpi], AS_HELP_STRING([--enable-mpi], [Build with MPI support]))
AS_IF([test "x$enable_mpi" = "xyes"], [LX_FIND_MPI])



dnl Boost
BOOST_REQUIRE([1.54.0])
BOOST_SERIALIZATION
AS_IF([test "x$enable_mpi" = "xyes"], [BOOST_MPI])
BOOST_PROGRAM_OPTIONS

AM_CONDITIONAL([HAVE_MPI], [test "x$have_CXX_mpi" == xyes])

AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_LDFLAGS])

AC_CONFIG_FILES([Makefile \
                 src/Makefile \
                 examples/Makefile])\
AC_OUTPUT

