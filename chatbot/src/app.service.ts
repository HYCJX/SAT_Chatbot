import axios from 'axios';

import { UserResponse } from './types';

const http = axios.create({
  baseURL: 'http://localhost:8000/',
  headers: {
    'Content-type': 'application/json',
  },
});

const postUserResponse = (userResponse: UserResponse) => {
  return http.post<UserResponse>(`/user_response`, userResponse);
};

const AppService = { postUserResponse };

export default AppService;
