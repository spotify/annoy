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


namespace detail {
  struct DataMapping
  {
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
      return Mapping{ nullptr, 0 };
    }
    off_t size = lseek(fd, 0, SEEK_END);
    void *mmaped = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mmaped == MAP_FAILED) {
      return Mapping{ nullptr, 0 };
    }
    if (need_mlock) {
      mlock(mmaped, size);
    }
    return Mapping{ mmaped, (size_t)size };
  }

  void unmap(const Mapping & mapping) {
    if (mapping.data) {
      munmap(const_cast<void*>(mapping.data), mapping.size);
    }
  }
};


class HugePagesDataMapper {
public:
    typedef detail::DataMapping Mapping;

public:
  Mapping map(const char* filename, bool need_mlock) {
    int fd = open(filename, O_RDONLY, (int)0400);
    if (fd == -1) {
      return Mapping{ nullptr, 0 };
    }

    Mapping mapping;
    mapping.size = lseek(fd, 0, SEEK_END);
    mapping.data = mmap(0, mapping.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);
    if (mapping.data == MAP_FAILED) {
      return Mapping{ nullptr, 0 };
    }
    lseek(fd, 0, SEEK_SET);

    size_t bytes_left = mapping.size;
    void *bytes = const_cast<void *>(mapping.data);
    while (bytes_left) {
      ssize_t count = read(fd, bytes, bytes_left);
      if (count <= 0) {
        return Mapping{ nullptr, 0 };
      }
      bytes = static_cast<char *>(bytes) + count;
      bytes_left -= count;
    }
    close(fd);

    return mapping;
  }

  void unmap(const Mapping & mapping) {
    if (mapping.data) {
      munmap(const_cast<void*>(mapping.data), mapping.size);
    }
  }
};

