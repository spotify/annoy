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

// WARNING! you need to enable at least SSSE3 instructions for this file! (-mssse3)
// or AVX2 (-mavx2) to gain better speed
#pragma once

#include <emmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>


#if not defined(__SSSE3__)
# error "You need to enable SSSE3 instructions for this file or higher!(-mssse3)"
#endif

#ifndef NO_MANUAL_VECTORIZATION
# if defined(__AVX2__)
#   define USE_AVX2
# endif
#endif

namespace Annoy {

#if defined(USE_AVX2)

inline void pack_float_vector_i16( float const *__restrict__ x, uint16_t *__restrict__ out, uint32_t d )
{
  __m256 mm1 = _mm256_set1_ps(32767.f);
  while( d >= 16  )
  {
      __m256 a = _mm256_loadu_ps(x);
      __m256 b = _mm256_loadu_ps(x + 8);
      __m256i ai = _mm256_cvtps_epi32(_mm256_mul_ps(a, mm1));
      __m256i bi = _mm256_cvtps_epi32(_mm256_mul_ps(b, mm1));
      _mm256_storeu_si256((__m256i*)out, _mm256_packs_epi32(ai, bi));
      x += 16;
      out += 16;
      d -= 16;
  }

  if( d )
  {
    __m128 m1 = _mm_set1_ps(32767.f);
    __m128 a = _mm_loadu_ps(x);
    __m128 b = _mm_loadu_ps(x + 4);
    __m128i ai = _mm_cvtps_epi32(_mm_mul_ps(a, m1));
    __m128i bi = _mm_cvtps_epi32(_mm_mul_ps(b, m1));
    _mm_store_si128((__m128i*)out, _mm_packs_epi32(ai, bi));
  }
}

inline void decode_vector_i16_f32( uint16_t const *__restrict__ in, float *__restrict__ out, uint32_t d )
{
  __m256 mm1 = _mm256_set1_ps(1.f / 32767.f);
  while( d >= 16  )
  {
      __m256i s  = _mm256_loadu_si256( (__m256i const*)(in) );
      __m256i ai = _mm256_srai_epi32(_mm256_unpacklo_epi16(s, s), 16);
      __m256i bi = _mm256_srai_epi32(_mm256_unpackhi_epi16(s, s), 16);
      __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(ai), mm1);
      __m256 b = _mm256_mul_ps(_mm256_cvtepi32_ps(bi), mm1);
      _mm256_storeu_ps(out, a);
      _mm256_storeu_ps(out + 8, b);
      in += 16;
      out += 16;
      d -= 16;
  }
  if( d )
  {
      __m128 m1 = _mm_set1_ps(1.f / 32767.f);
      __m128i s  = _mm_loadu_si128( (__m128i const*)(in) );
      __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
      __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
      __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
      __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
      _mm_storeu_ps(out, a);
      _mm_storeu_ps(out + 4, b);
  }
}

inline float decode_and_dot_i16_f32( uint16_t const *__restrict__ in, float const *__restrict__ y, uint32_t d )
{
  float sum;
  __m256 mm1 = _mm256_set1_ps(1.f / 32767.f);
  __m256 msum1 = _mm256_setzero_ps(), msum2 = _mm256_setzero_ps();
  __m256 mx, my;
  while( d >= 16  )
  {
      // every step decoded into 16 floats
      // sadly but we need to use here unaligned load
      // due to 16 byte offset for vector, not 32 byte!
      __m256i s  = _mm256_lddqu_si256( (__m256i const*)(in) );
      __m256i ai = _mm256_srai_epi32(_mm256_unpacklo_epi16(s, s), 16);
      __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(ai), mm1);
      mx = _mm256_load_ps(y);
      __m256i bi = _mm256_srai_epi32(_mm256_unpackhi_epi16(s, s), 16);
      msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (a, mx));
      __m256 b = _mm256_mul_ps(_mm256_cvtepi32_ps(bi), mm1);
      my = _mm256_load_ps(y + 8);
      msum2 = _mm256_add_ps (msum2, _mm256_mul_ps (b, my));
      in += 16;
      y += 16;
      d -= 16;
  }
  msum1 = _mm256_add_ps(msum1, msum2);
  // now sum of 8
  msum1 = _mm256_hadd_ps (msum1, msum1);
  msum1 = _mm256_hadd_ps (msum1, msum1);
  // now 0 and 4 left
  sum = _mm_cvtss_f32 (_mm256_castps256_ps128(msum1)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(msum1, 1));

  if( d )
  {
    // every step decoded into 8 floats
    // use here slow _mm_dp_ps instruction since is no loop here
    __m128 m1 = _mm_set1_ps(1.f / 32767.f);
    __m128i s  = _mm_load_si128( (__m128i const*)(in) );
    __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
    __m128 mx = _mm_load_ps (y);
    __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
    __m128 mdot = _mm_dp_ps (a, mx, 0xff);
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
    __m128 my = _mm_load_ps (y + 4);
    __m128 mdot2 = _mm_dp_ps (b, my, 0xff);
    __m128 msum = _mm_add_ps (mdot, mdot2);
    sum += _mm_cvtss_f32(msum);
  }

  return sum;
}

inline float decode_and_euclidean_distance_i16_f32( uint16_t const *__restrict__ in, float const *__restrict__ y, uint32_t d )
{

  float sum;
  __m256 mm1 = _mm256_set1_ps(1.f / 32767.f);
  __m256 msum1 = _mm256_setzero_ps(), msum2 = _mm256_setzero_ps();
  __m256 mx, my;
  while( d >= 16  )
  {
      // every step decoded into 16 floats
      // sadly but we need to use here unaligned load
      // due to 16 byte offset for vector, not 32 byte!
      __m256i s  = _mm256_lddqu_si256( (__m256i const*)(in) );
      __m256i ai = _mm256_srai_epi32(_mm256_unpacklo_epi16(s, s), 16);
      __m256 a = _mm256_mul_ps(_mm256_cvtepi32_ps(ai), mm1);
      mx = _mm256_loadu_ps(y);
      __m256i bi = _mm256_srai_epi32(_mm256_unpackhi_epi16(s, s), 16);
      __m256 d1 = _mm256_sub_ps (a, mx);
      msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (d1, d1));
      __m256 b = _mm256_mul_ps(_mm256_cvtepi32_ps(bi), mm1);
      my = _mm256_loadu_ps(y + 8);
      __m256 d2 = _mm256_sub_ps (b, my);
      msum2 = _mm256_add_ps (msum2, _mm256_mul_ps (d2, d2));
      in += 16;
      y += 16;
      d -= 16;
  }
  msum1 = _mm256_add_ps(msum1, msum2);
  // now sum of 8
  msum1 = _mm256_hadd_ps (msum1, msum1);
  msum1 = _mm256_hadd_ps (msum1, msum1);
  // now 0 and 4 left
  sum = _mm_cvtss_f32 (_mm256_castps256_ps128(msum1)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(msum1, 1));

  // here can be 0/8 left, so do check and calc tail if exists
  if( d )
  {
    __m128 m1 = _mm_set1_ps(1.f / 32767.f);
    __m128 msum1 = _mm_setzero_ps();
    __m128 mx, my;
    __m128i s  = _mm_load_si128( (__m128i const*)(in) );
    __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
    mx = _mm_load_ps (y);
    __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
    __m128 d1 = _mm_sub_ps (a, mx);
    msum1 = _mm_add_ps (msum1, _mm_mul_ps (d1, d1));
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
    my = _mm_load_ps (y + 4);
    __m128 d2 = _mm_sub_ps (b, my);
    msum1 = _mm_add_ps (msum1, _mm_mul_ps (d2, d2));
    msum1 = _mm_hadd_ps (msum1, msum1);
    msum1 = _mm_hadd_ps (msum1, msum1);
    sum += _mm_cvtss_f32 (msum1);
  }


  return sum;
}

#else

inline void pack_float_vector_i16( float const *__restrict__ x, uint16_t *__restrict__ out, uint32_t d )
{
  __m128 m1 = _mm_set1_ps(32767.f);
  for( uint32_t i = 0; i < d; i += 8 )
  {
    __m128 a = _mm_loadu_ps(x + i);
    __m128 b = _mm_loadu_ps(x + i + 4);
    __m128i ai = _mm_cvtps_epi32(_mm_mul_ps(a, m1));
    __m128i bi = _mm_cvtps_epi32(_mm_mul_ps(b, m1));
    __m128i *op = (__m128i*)(out + i);
    // temporal store???
    // _mm_stream_si128(op, _mm_packs_epi32(ai, bi));
    _mm_store_si128(op, _mm_packs_epi32(ai, bi));
  }
}

inline void decode_vector_i16_f32( uint16_t const *__restrict__ in, float *__restrict__ out, uint32_t d )
{
  __m128 m1 = _mm_set1_ps(1.f / 32767.f);
  // every step decoded into 8 float at once!
  for( uint32_t i = 0; i < d; i += 8 )
  {
    __m128i s  = _mm_loadu_si128( (__m128i const*)(in + i) );
    __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
    __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
    _mm_storeu_ps(out + i, a);
    _mm_storeu_ps(out + i + 4, b);
  }
}

inline float decode_and_dot_i16_f32( uint16_t const *__restrict__ in, float const *__restrict__ y, uint32_t d )
{
  __m128 m1 = _mm_set1_ps(1.f / 32767.f);
  __m128 msum1 = _mm_setzero_ps(), msum2 = _mm_setzero_ps();
  __m128 mx, my;
  // every step decoded into 8 float at once!
  uint32_t i = 0;
  do
  {
    __m128i s  = _mm_load_si128( (__m128i const*)(in + i) );
    __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
    mx = _mm_load_ps (y + i + 0);
    __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
    msum1 = _mm_add_ps (msum1, _mm_mul_ps (a, mx));
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
    my = _mm_load_ps (y + i + 4);
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (b, my));
    i += 8;
  }
  while( i < d );

  msum1 = _mm_add_ps(msum1, msum2);

  msum1 = _mm_hadd_ps (msum1, msum1);
  msum1 = _mm_hadd_ps (msum1, msum1);
  return  _mm_cvtss_f32 (msum1);
}

inline float decode_and_euclidean_distance_i16_f32( uint16_t const *__restrict__ in, float const *__restrict__ y, uint32_t d )
{
  __m128 m1 = _mm_set1_ps(1.f / 32767.f);
  __m128 msum1 = _mm_setzero_ps(), msum2 = _mm_setzero_ps();
  __m128 mx, my;
  // every step decoded into 8 float at once!
  uint32_t i = 0;
  do
  {
    __m128i s  = _mm_load_si128( (__m128i const*)(in + i) );
    __m128i ai = _mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(ai), m1);
    mx = _mm_load_ps (y + i + 0);
    __m128i bi = _mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16);
    __m128 d1 = _mm_sub_ps (a, mx);
    msum1 = _mm_add_ps (msum1, _mm_mul_ps (d1, d1));
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(bi), m1);
    my = _mm_load_ps (y + i + 4);
    __m128 d2 = _mm_sub_ps (b, my);
    msum2 = _mm_add_ps (msum2, _mm_mul_ps (d2, d2));
    i += 8;
  }
  while( i < d );

  msum1 = _mm_add_ps(msum1, msum2);

  msum1 = _mm_hadd_ps (msum1, msum1);
  msum1 = _mm_hadd_ps (msum1, msum1);
  return  _mm_cvtss_f32 (msum1);
}

#endif

} // namespace Annoy
