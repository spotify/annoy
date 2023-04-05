// Copyright (c) 2013 Spotify AB
// Copyright (c) 2018 viper.craft@gmail.com
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace Annoy {

namespace detail {
  struct DataMapping
  {
    DataMapping( void const *d, size_t s )
      : data(d)
      , size(s)
      {}

    DataMapping() = default;
    
    void const *data = nullptr;
    size_t size = 0;
  };

} // namespace detail


class MMapDataMapper {
public:
    typedef detail::DataMapping Mapping;

public:
  Mapping map(const char* filename, bool need_mlock) {
    int fd = open(filename, O_RDONLY, (int)0400);
    if (fd == -1) {
      return Mapping();
    }

    struct stat fd_stat;
    if (fstat(fd, &fd_stat)) {
        return Mapping();
    }

    void *mmaped = mmap(0, fd_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mmaped == MAP_FAILED) {
      return Mapping();
    }
#if defined(MADV_DONTDUMP)
    // Exclude from a core dump those pages
    madvise(mmaped, fd_stat.st_size, MADV_DONTDUMP);
#endif

    if (need_mlock) {
      mlock(mmaped, fd_stat.st_size);
    }

    return Mapping(mmaped, (size_t)fd_stat.st_size);
  }

  void unmap(const Mapping & mapping) {
    if (mapping.data) {
      munmap(const_cast<void*>(mapping.data), mapping.size);
    }
  }
};

// Anonymous HugeTLB mapping.
// Valid only on some Linux based-systems
#if defined(MAP_HUGETLB)
// Allocates memory from pool and populates it by file data from disk.
// It may be better to consider HugeTLB FS instead. In that case you will get instant data loading.
class HugePagesDataMapper {
public:
    typedef detail::DataMapping Mapping;

public:
  Mapping map(const char* filename, bool need_mlock) {
    int fd = open(filename, O_RDONLY, (int)0400);
    if (fd == -1) {
      return Mapping();
    }

    struct stat fd_stat;
    if (fstat(fd, &fd_stat)) {
        return Mapping();
    }

    Mapping mapping;
    mapping.size = fd_stat.st_size;
    mapping.data = mmap(0, mapping.size, PROT_READ | PROT_WRITE, 
                        MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);
    if (mapping.data == MAP_FAILED) {
      return Mapping();
    }

    size_t bytes_left = mapping.size;
    void *bytes = const_cast<void *>(mapping.data);
    while (bytes_left) {
      ssize_t count = read(fd, bytes, bytes_left);
      if (count <= 0) {
        return Mapping();
      }
      bytes = static_cast<char *>(bytes) + count;
      bytes_left -= count;
    }
    close(fd);

#if defined(MADV_DONTDUMP)
    // Exclude from a core dump those pages
    madvise(const_cast<void*>(mapping.data), fd_stat.st_size, MADV_DONTDUMP);
#endif

    return mapping;
  }

  void unmap(const Mapping & mapping) {
    if (mapping.data) {
      munmap(const_cast<void*>(mapping.data), mapping.size);
    }
  }
};

#endif

} // namespace Annoy
