AM_CPPFLAGS = -I$(top_srcdir)/src @AM_CXXFLAGS@ $(MPI_CXXFLAGS) $(BOOST_CPPFLAGS) -ansi -pipe -O3 -march=native -ffast-math -pedantic
AM_LDFLAGS = $(BOOST_SERIALIZATION_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LDFLAGS) $(MPI_CXXLDFLAGS) $(BOOST_MPI_LDFLAGS)
LDADD = $(top_srcdir)/src/libauryn.la $(AM_LDFLAGS)

bin_PROGRAMS = sim_background \
			   sim_brunel2k \
			   sim_coba_benchmark \
			   sim_epsp \
			   sim_isp_big \
			   sim_poisson \
			   sim_bg_static \
			   sim_brunel2k_pl \
			   sim_dense \
			   sim_epsp_stp \
			   sim_isp_orig


sim_background_SOURCES = sim_background.cpp
sim_brunel2k_SOURCES = sim_brunel2k.cpp
sim_coba_benchmark_SOURCES = sim_coba_benchmark.cpp
sim_epsp_SOURCES = sim_epsp.cpp
sim_isp_big_SOURCES = sim_isp_big.cpp
sim_poisson_SOURCES = sim_poisson.cpp
sim_bg_static_SOURCES = sim_bg_static.cpp
sim_brunel2k_pl_SOURCES = sim_brunel2k_pl.cpp
sim_dense_SOURCES = sim_dense.cpp
sim_epsp_stp_SOURCES = sim_epsp_stp.cpp
sim_isp_orig_SOURCES = sim_isp_orig.cpp
