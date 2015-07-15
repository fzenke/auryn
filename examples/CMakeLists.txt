ADD_EXECUTABLE( sim_background sim_background.cpp )
TARGET_LINK_LIBRARIES( sim_background libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_bg_lowpass sim_bg_lowpass.cpp )
TARGET_LINK_LIBRARIES( sim_bg_lowpass libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_bg_static sim_bg_static.cpp )
TARGET_LINK_LIBRARIES( sim_bg_static libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_brunel2k sim_brunel2k.cpp )
TARGET_LINK_LIBRARIES( sim_brunel2k libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_brunel2k_pl sim_brunel2k_pl.cpp )
TARGET_LINK_LIBRARIES( sim_brunel2k_pl libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_coba_benchmark sim_coba_benchmark.cpp )
TARGET_LINK_LIBRARIES( sim_coba_benchmark libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_dense sim_dense.cpp )
TARGET_LINK_LIBRARIES( sim_dense libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_epsp sim_epsp.cpp )
TARGET_LINK_LIBRARIES( sim_epsp libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_epsp_stp sim_epsp_stp.cpp )
TARGET_LINK_LIBRARIES( sim_epsp_stp libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_isp_big sim_isp_big.cpp )
TARGET_LINK_LIBRARIES( sim_isp_big libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_isp_orig sim_isp_orig.cpp )
TARGET_LINK_LIBRARIES( sim_isp_orig libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

ADD_EXECUTABLE( sim_poisson sim_poisson.cpp )
TARGET_LINK_LIBRARIES( sim_poisson libauryn ${MPI_C_LIBRARIES} ${Boost_LIBRARIES} )

