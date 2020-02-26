// -*- c++ -*-
/****************************************************************
 ****
 **** This file is part of the prototype implementation of
 **** the Integrative Model for Parallelism
 ****
 **** copyright Victor Eijkhout 2014-6
 ****
 **** doxygen.cxx : this file fixes the has-a relationship in class diagrams in doxygen
 ****     also has lots of documentation
 ****
 ****************************************************************/

namespace std
{
  template<class T> class vector { public: T *_0_to_N; };
}

/*! \page objects The object hierarchy

  Here are the main objects:
  - \ref indexstruct : a set of indices. There are various derived classes.
  - \ref parallel_indexstruct : mapping from a linear set of processors
    to \ref indexstruct objects.
  - \ref parallel_structure : multi-dimensional cartesian version of parallel_indexstruct:
    there is a \ref parallel_indexstruct for each dimension.
  - \ref distribution : a \ref parallel_structure with adornments, such as \ref processor_mask
    and orthogonal dimension and \ref communicator
  - \ref object : data allocated according to a \ref distribution
  - \ref kernel : this describes an operation taking one object and deliving another.
  - \ref task : a \ref kernel on a specific processor.

  \section domain Domains

  Domains are the subdivisions of the global domain. 
  - in MPI, a processor likely has only one domain: the MPI rank.
  - in OMP, there is only one NUMA domain, and it contains all domains: the OMP thread numbers.

  You find the local number of a domain with \ref decomposition::get_domain_local_number,
  which throws an exception if the domain is not part of the NUMA domain.

  There is data associated with a domain, which gets set by \ref object::register_data_on_domain_number.

  \section multid Multi-dimensional support
 */
