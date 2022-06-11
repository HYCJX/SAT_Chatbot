import { faPaperPlane } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import React, { useState } from 'react';

import Chats from '../Chats';
import './Chatbot.scss';

interface ResponseBotObject {
  message: string;
  options?: string[];
}

const Chatbot: React.FC = () => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [step, setStep] = useState<number>(0);
  const [editingUserResponse, setEditingUserResponse] = useState<string>('');
  const [botResponse, setBotResponse] = useState<ResponseBotObject>({ message: '' });
  const [submittedUserResponse, setSubmittedSendUserResponse] = useState<string>('');

  // setting next step when there's response and option click
  const setNextStep = (response: string) => {
    setStep((step) => step + 1);
    setSubmittedSendUserResponse(response);
    let res = { message: '' };
    setBotResponse(res);
    setEditingUserResponse('');
  };

  const optionClick = (e: React.MouseEvent<HTMLElement>) => {
    let option = e.currentTarget.dataset.id;
    if (option) {
      setNextStep(option);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditingUserResponse(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setNextStep(editingUserResponse);
  };

  return (
    <div className="chat-container">
      <Chats
        userResponse={submittedUserResponse}
        botResponse={botResponse}
        optionClick={optionClick}
      />
      <form onSubmit={(e) => handleSubmit(e)} className="form-container">
        <input onChange={(e) => handleInputChange(e)} value={editingUserResponse}></input>
        <button>
          <FontAwesomeIcon icon={faPaperPlane} />
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
