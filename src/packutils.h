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

// WARNING! you need to enable SSSE3 instructions for this file! (-mssse3)

#pragma once

#include <emmintrin.h>
#include <pmmintrin.h>
#include <stdint.h>

static float const _15BITS_MULT = 32767.f, _15BITS_DIVISOR = 1.f / _15BITS_MULT;

void pack_float_vector_i16( float const *__restrict__ x, uint16_t *__restrict__ out, uint32_t const d )
{
  __m128 m1 = _mm_set1_ps(_15BITS_MULT);
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

void decode_vector_i16_f32( uint16_t const *__restrict__ in, float *__restrict__ out, uint32_t const d )
{
  __m128 m1 = _mm_set1_ps(_15BITS_DIVISOR);
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

float decode_and_dot_i16_f32( uint16_t const *__restrict__ in, float const *__restrict__ y, uint32_t const d )
{
  // fetch couple of lines to trigger hardware prefetch
  __builtin_prefetch((uint8_t const*)in + 64);
  __builtin_prefetch((uint8_t const*)in + 128);
  __builtin_prefetch((uint8_t const*)in + 192);
  __m128 m1 = _mm_set1_ps(_15BITS_DIVISOR);
  __m128 msum1 = _mm_setzero_ps(), msum2 = _mm_setzero_ps();
  __m128 mx, my;    
  // every step decoded into 8 float at once!
  for( uint32_t i = 0; i < d; i += 8 )
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
  }

  msum1 = _mm_add_ps(msum1, msum2);

  msum1 = _mm_hadd_ps (msum1, msum1);
  msum1 = _mm_hadd_ps (msum1, msum1);
  return  _mm_cvtss_f32 (msum1);
}
